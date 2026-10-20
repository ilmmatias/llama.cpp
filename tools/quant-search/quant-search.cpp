#include "common.h"
#include "imatrix-loader.h"
#include "llama.h"
#include "ggml-cpp.h"

#include "../../src/llama-ext.h"
#include "../../src/llama-mmap.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <clocale>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <regex>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

struct tensor_rule {
    std::regex pattern;
    std::vector<ggml_type> types;
};

struct search_params {
    std::string model;
    std::string imatrix;
    std::vector<std::string> eval_recipes;
    std::string calibration_file;
    std::string eval_cache;
    std::string global_recipe;
    size_t response_rank = 64;
    std::vector<std::regex> global_tensors;
    uint32_t n_ctx = 512;
    uint32_t n_batch = 32;
    size_t n_chunks = 1;
    std::vector<ggml_type> types;
    std::vector<tensor_rule> rules;
    std::vector<std::regex> exclude_tensors;
    std::vector<std::regex> exclude_bpw;
    size_t max_buffer_size = 64 * 1024 * 1024;
    size_t max_nodes = 100000;
    long double target_bpw = 0.0;
    int n_threads = std::max(1u, std::thread::hardware_concurrency());
    bool dry_run = false;
    bool allow_requantize = false;
};

struct input_file {
    std::string path;
    ggml_context_ptr meta;
    gguf_context_ptr gguf;
};

struct candidate {
    ggml_type type;
    size_t size;
    double error = 0.0;
    double error_weighted = 0.0;
};

struct tensor_plan {
    const ggml_tensor * tensor;
    size_t file;
    size_t offset;
    bool counted;
    std::vector<candidate> candidates;
};

static void usage(const char * executable) {
    fprintf(stderr, "usage: %s [options] model.gguf type[,type...]\n\n", executable);
    fprintf(stderr, "Measure per-tensor costs and errors, or select types under a BPW limit.\n\n");

    fprintf(stderr, "  -h, --help                  show this help message\n");
    fprintf(stderr, "  --imatrix FILE              use activation statistics from llama-imatrix\n");
    fprintf(stderr, "  --tensor-type PATTERN=TYPES override candidate types (first matching regex wins)\n");
    fprintf(stderr, "  --exclude-tensors PATTERN   preserve matching tensors without searching them\n");
    fprintf(stderr, "  --exclude-bpw PATTERN       omit matching tensors from BPW accounting\n");
    fprintf(stderr, "  --target-bpw N              write a llama-quantize tensor-type file to stdout\n");
    fprintf(stderr, "  --max-nodes N               allocation search limit (default: 100000)\n");
    fprintf(stderr, "  --max-buffer-size MiB       limit tensor buffers (default: 64 MiB)\n");
    fprintf(stderr, "  -t, --threads N             number of quantization threads\n");
    fprintf(stderr, "  --allow-requantize          allow conversion from quantized source weights\n");
    fprintf(stderr, "  --dry-run                   report types and sizes without reading weights\n");
    fprintf(stderr, "  --eval-recipe FILE          evaluate a tensor-type recipe (repeatable)\n");
    fprintf(stderr, "  --calibration-file FILE     text used for global recipe evaluation\n");
    fprintf(stderr, "  --eval-cache PREFIX         temporary disk cache prefix for evaluation\n");
    fprintf(stderr, "  --ctx-size N                evaluation context size (default: 512)\n");
    fprintf(stderr, "  --batch-size N              evaluation batch size (default: 32)\n");
    fprintf(stderr, "  --chunks N                  evaluation context chunks (default: 1)\n");
    fprintf(stderr, "  --global-search FILE        refine a feasible initial recipe using global KL\n");
    fprintf(stderr, "  --response-rank N           even Fisher sketch size per token (default: 64)\n");
    fprintf(stderr, "  --global-tensors PATTERN    only refine matching tensors in the initial recipe\n");
}

static size_t checked_add(size_t a, size_t b) {
    if (b > SIZE_MAX - a) {
        throw std::runtime_error("size overflow");
    }

    return a + b;
}

static size_t checked_mul(size_t a, size_t b) {
    if (a != 0 && b > SIZE_MAX / a) {
        throw std::runtime_error("size overflow");
    }

    return a * b;
}

static std::vector<ggml_type> parse_types(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) { return std::tolower(c); });

    std::vector<ggml_type> types;
    for (const auto & name : string_split<std::string>(text, ',')) {
        ggml_type found = GGML_TYPE_COUNT;

        for (int i = 0; i < GGML_TYPE_COUNT; ++i) {
            const auto type = (ggml_type) i;
            std::string type_name = ggml_type_name(type);
            std::transform(type_name.begin(), type_name.end(), type_name.begin(), [](unsigned char c) { return std::tolower(c); });
            const auto * traits = ggml_get_type_traits(type);
            if (name == type_name && (type == GGML_TYPE_F32 || traits->to_float)) {
                found = type;
                break;
            }
        }

        if (found == GGML_TYPE_COUNT) {
            throw std::runtime_error("unsupported quantization type: " + name);
        }

        if (std::find(types.begin(), types.end(), found) == types.end()) {
            types.push_back(found);
        }
    }

    if (types.empty() || text.empty() || text.back() == ',') {
        throw std::runtime_error("empty candidate type list");
    }

    return types;
}

static size_t parse_positive(const std::string & text) {
    if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos) {
        throw std::runtime_error("expected a positive integer: " + text);
    }

    const auto n = std::stoull(text);
    if (n == 0 || n > SIZE_MAX) {
        throw std::runtime_error("integer out of range: " + text);
    }

    return (size_t) n;
}

static search_params parse_params(int argc, char ** argv) {
    search_params params;
    std::vector<std::string> positional;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];

        if (arg == "--dry-run") {
            params.dry_run = true;
        } else if (arg == "--allow-requantize") {
            params.allow_requantize = true;
        } else if (arg == "--imatrix" || arg == "--tensor-type" || arg == "--exclude-tensors" ||
                   arg == "--exclude-bpw" || arg == "--max-buffer-size" || arg == "--target-bpw" ||
                   arg == "--max-nodes" || arg == "-t" || arg == "--threads" ||
                   arg == "--eval-recipe" || arg == "--calibration-file" || arg == "--eval-cache" ||
                   arg == "--ctx-size" || arg == "--batch-size" || arg == "--chunks" ||
                   arg == "--global-search" || arg == "--response-rank" || arg == "--global-tensors") {
            if (++i == argc) {
                throw std::runtime_error("missing value for " + arg);
            }

            const std::string value = argv[i];
            if (arg == "--imatrix") {
                params.imatrix = value;
            } else if (arg == "--eval-recipe") {
                params.eval_recipes.push_back(value);
            } else if (arg == "--calibration-file") {
                params.calibration_file = value;
            } else if (arg == "--eval-cache") {
                params.eval_cache = value;
            } else if (arg == "--global-search") {
                params.global_recipe = value;
            } else if (arg == "--global-tensors") {
                params.global_tensors.emplace_back(value);
            } else if (arg == "--response-rank") {
                params.response_rank = parse_positive(value);
                if (params.response_rank < 2 || params.response_rank % 2) {
                    throw std::runtime_error("--response-rank requires a positive even size of at least two");
                }
            } else if (arg == "--ctx-size" || arg == "--batch-size" || arg == "--chunks") {
                const size_t n = parse_positive(value);
                if (n > INT32_MAX) {
                    throw std::runtime_error("evaluation size out of range");
                }

                if (arg == "--ctx-size") {
                    params.n_ctx = (uint32_t) n;
                } else if (arg == "--batch-size") {
                    params.n_batch = (uint32_t) n;
                } else {
                    params.n_chunks = n;
                }
            } else if (arg == "--tensor-type") {
                const size_t sep = value.find('=');
                if (sep == std::string::npos || sep == 0) {
                    throw std::runtime_error("expected PATTERN=TYPE[,TYPE...]");
                }

                params.rules.push_back({std::regex(value.substr(0, sep)), parse_types(value.substr(sep + 1))});
            } else if (arg == "--exclude-tensors") {
                params.exclude_tensors.emplace_back(value);
            } else if (arg == "--exclude-bpw") {
                params.exclude_bpw.emplace_back(value);
            } else if (arg == "--max-buffer-size") {
                params.max_buffer_size = checked_mul(parse_positive(value), 1024 * 1024);
            } else if (arg == "--max-nodes") {
                params.max_nodes = parse_positive(value);
            } else if (arg == "--target-bpw") {
                size_t end = 0;
                params.target_bpw = std::stold(value, &end);
                if (end != value.size() || !std::isfinite(params.target_bpw) || params.target_bpw <= 0) {
                    throw std::runtime_error("expected a positive finite BPW limit");
                }
            } else {
                const size_t n = parse_positive(value);
                if (n > INT32_MAX) {
                    throw std::runtime_error("thread count out of range");
                }

                params.n_threads = (int) n;
            }
        } else if (!arg.empty() && arg[0] == '-') {
            throw std::runtime_error("unknown option: " + arg);
        } else {
            positional.push_back(arg);
        }
    }

    if (positional.size() != 2) {
        throw std::runtime_error("expected a model path and a candidate type list");
    }

    params.model = positional[0];
    params.types = parse_types(positional[1]);

    if (params.dry_run && params.target_bpw > 0) {
        throw std::runtime_error("--target-bpw requires measured errors; remove --dry-run");
    }

    if (!params.eval_recipes.empty() && (params.calibration_file.empty() || params.eval_cache.empty() || params.dry_run || params.target_bpw > 0)) {
        throw std::runtime_error("--eval-recipe requires --calibration-file and --eval-cache; cannot combine it with --target-bpw or --dry-run");
    }

    if (!params.global_recipe.empty() && (params.target_bpw <= 0 || params.calibration_file.empty() || params.eval_cache.empty() || params.dry_run || !params.eval_recipes.empty())) {
        throw std::runtime_error("--global-search requires --target-bpw, --calibration-file, and --eval-cache; cannot combine it with --eval-recipe or --dry-run");
    }

    if (params.eval_recipes.empty() && params.global_recipe.empty() && (!params.calibration_file.empty() || !params.eval_cache.empty())) {
        throw std::runtime_error("evaluation files require --eval-recipe or --global-search");
    }

    if (params.global_recipe.empty() && !params.global_tensors.empty()) {
        throw std::runtime_error("--global-tensors requires --global-search");
    }

    return params;
}

static bool matches(const std::vector<std::regex> & patterns, const char * name) {
    for (const auto & pattern : patterns) {
        if (std::regex_search(name, pattern)) {
            return true;
        }
    }

    return false;
}

static uint16_t split_value(const gguf_context * ctx, const char * key, uint16_t fallback) {
    const int64_t id = gguf_find_key(ctx, key);
    if (id < 0) {
        return fallback;
    }

    if (gguf_get_kv_type(ctx, id) != GGUF_TYPE_UINT16) {
        throw std::runtime_error(std::string("expected uint16 metadata: ") + key);
    }

    return gguf_get_val_u16(ctx, id);
}

static std::vector<input_file> open_inputs(const std::string & path) {
    std::vector<input_file> files;
    std::vector<char> prefix(path.size() + 32);
    std::vector<char> split_path(path.size() + 32);
    uint16_t n_split = 1;

    for (uint16_t i = 0; i < n_split; ++i) {
        input_file file;
        file.path = path;

        if (i > 0) {
            if (!llama_split_path(split_path.data(), split_path.size(), prefix.data(), i, n_split)) {
                throw std::runtime_error("invalid split path");
            }

            file.path = split_path.data();
        }

        ggml_context * meta = nullptr;
        file.gguf.reset(gguf_init_from_file(file.path.c_str(), {true, &meta}));
        file.meta.reset(meta);
        if (!file.gguf) {
            throw std::runtime_error("cannot read GGUF: " + file.path);
        }

        if (i == 0) {
            n_split = split_value(file.gguf.get(), "split.count", 1);
            if (n_split == 0 || split_value(file.gguf.get(), "split.no", 0) != 0) {
                throw std::runtime_error("open the first GGUF shard");
            }

            if (n_split > 1 && !llama_split_prefix(prefix.data(), prefix.size(), path.c_str(), 0, n_split)) {
                throw std::runtime_error("invalid split filename: " + path);
            }
        }

        if (split_value(file.gguf.get(), "split.count", 1) != n_split || split_value(file.gguf.get(), "split.no", 0) != i) {
            throw std::runtime_error("inconsistent split metadata: " + file.path);
        }

        files.push_back(std::move(file));
    }

    return files;
}

static std::vector<tensor_plan> plan_tensors(const search_params & params, const std::vector<input_file> & files,
                                            const common_imatrix & imatrix) {
    std::vector<tensor_plan> plans;
    std::set<std::string> names;
    const auto qparams = llama_model_quantize_default_params();

    const auto * metadata = files[0].gguf.get();
    const int64_t arch_id = gguf_find_key(metadata, "general.architecture");
    if (arch_id < 0 || gguf_get_kv_type(metadata, arch_id) != GGUF_TYPE_STRING) {
        throw std::runtime_error("missing general.architecture string");
    }

    const char * architecture = gguf_get_val_str(metadata, arch_id);

    for (size_t f = 0; f < files.size(); ++f) {
        const auto * ctx = files[f].gguf.get();
        std::ifstream in(files[f].path, std::ios::binary | std::ios::ate);
        if (!in || in.tellg() < 0) {
            throw std::runtime_error("cannot open " + files[f].path);
        }

        const uint64_t file_size = (uint64_t) in.tellg();

        for (int64_t i = 0; i < gguf_get_n_tensors(ctx); ++i) {
            const char * name = gguf_get_tensor_name(ctx, i);
            if (!names.insert(name).second) {
                throw std::runtime_error(std::string("duplicate tensor: ") + name);
            }

            const auto * tensor = ggml_get_tensor(files[f].meta.get(), name);
            const size_t offset = checked_add(gguf_get_data_offset(ctx), gguf_get_tensor_offset(ctx, i));
            if (offset > file_size || ggml_nbytes(tensor) > file_size - offset) {
                throw std::runtime_error(std::string("tensor outside file: ") + name);
            }

            tensor_plan plan{tensor, f, offset, !matches(params.exclude_bpw, name), {}};
            const std::vector<ggml_type> * types = &params.types;
            bool overridden = false;

            for (const auto & rule : params.rules) {
                if (std::regex_search(name, rule.pattern)) {
                    types = &rule.types;
                    overridden = true;
                    break;
                }
            }

            const bool eligible = llama_quant_tensor_allows_quantization(architecture, &qparams, tensor);
            const bool excluded = matches(params.exclude_tensors, name);
            if (!eligible || excluded) {
                if (overridden && !excluded && (types->size() != 1 || (*types)[0] != tensor->type)) {
                    throw std::runtime_error(std::string("tensor is not eligible for quantization: ") + name);
                }

                plan.candidates.push_back({tensor->type, ggml_nbytes(tensor)});
            } else {
                for (const auto type : *types) {
                    const auto * traits = ggml_get_type_traits(type);
                    const char * reason = nullptr;

                    if (tensor->ne[0] % traits->blck_size != 0) {
                        reason = "row is not a multiple of the block size";
                    } else if (type != tensor->type && imatrix.entries.count(name) == 0 &&
                               (ggml_quantize_requires_imatrix(type) || llama_quant_tensor_requires_imatrix(name, type, LLAMA_FTYPE_MOSTLY_Q4_K_M))) {
                        reason = "requires imatrix data";
                    }

                    if (reason) {
                        fprintf(stderr, "%s: skip %s: %s\n", name, ggml_type_name(type), reason);
                        continue;
                    }

                    const size_t row_size = checked_mul(tensor->ne[0] / traits->blck_size, traits->type_size);
                    plan.candidates.push_back({type, checked_mul(row_size, ggml_nrows(tensor))});
                }

                if (plan.candidates.empty()) {
                    throw std::runtime_error(std::string("no valid candidates for ") + name);
                }

                for (const auto & c : plan.candidates) {
                    if (c.type != tensor->type && ggml_is_quantized(tensor->type) && !params.allow_requantize) {
                        throw std::runtime_error(std::string("use --allow-requantize for ") + name);
                    }
                }
            }

            plans.push_back(std::move(plan));
        }
    }

    return plans;
}

static void to_float(ggml_type type, const void * data, float * out, int64_t n) {
    if (type == GGML_TYPE_F32) {
        memcpy(out, data, n * sizeof(float));
    } else {
        const auto * traits = ggml_get_type_traits(type);
        if (!traits->to_float) {
            throw std::runtime_error("cannot decode source type");
        }

        traits->to_float(data, out, n);
    }
}

struct tensor_progress {
    const char * name;
    size_t rows;
    size_t n_candidates;
    bool terminal = common_is_tty(stderr);
    int64_t started = ggml_time_us();
    int64_t updated = started;
    int width = 0;

    void report(ggml_type type, size_t candidate_index, size_t rows_done, long double fraction, bool force = false) {
        const int64_t now = ggml_time_us();
        if (!force && now - updated < 1000000) {
            return;
        }

        if (terminal) {
            fputc('\r', stderr);
        }

        const int written = fprintf(stderr, "%s: %6.2Lf%%, %s (%zu/%zu), rows %zu/%zu, %.1fs",
                                    name, 100 * fraction, ggml_type_name(type), candidate_index, n_candidates,
                                    rows_done, rows, (now - started) / 1e6);
        if (terminal) {
            for (int i = written; i < width; ++i) {
                fputc(' ', stderr);
            }

            width = std::max(0, written);
        } else {
            fputc('\n', stderr);
        }

        fflush(stderr);
        updated = now;
    }

    ~tensor_progress() {
        if (terminal && width > 0) {
            fputc('\n', stderr);
            fflush(stderr);
        }
    }
};

static bool measure_tensor(const search_params & params, const input_file & file, tensor_plan & plan,
                           const common_imatrix & imatrix, size_t tensor_index, size_t n_tensors, FILE * encoded_file = nullptr) {
    if (params.dry_run || (params.target_bpw > 0 && plan.candidates.size() == 1)) {
        return false;
    }

    const auto * tensor = plan.tensor;
    size_t n_converts = 0;
    size_t row_size_max = 0;
    for (const auto & c : plan.candidates) {
        n_converts += c.type != tensor->type;
        row_size_max = std::max(row_size_max, ggml_row_size(c.type, tensor->ne[0]));
    }

    if (n_converts == 0) {
        return false;
    }

    const size_t cols = tensor->ne[0];
    const size_t rows = ggml_nrows(tensor);
    const size_t row_size_src = ggml_row_size(tensor->type, cols);
    const size_t bytes_per_row = checked_add(checked_add(row_size_src, row_size_max), checked_mul(cols, 2 * sizeof(float)));
    if (bytes_per_row > params.max_buffer_size) {
        throw std::runtime_error(std::string(tensor->name) + ": one row needs " + std::to_string(bytes_per_row) + " buffer bytes");
    }

    const size_t slab_rows = std::min(rows, params.max_buffer_size / bytes_per_row);
    const int n_threads = (int) std::min(slab_rows, (size_t) params.n_threads);
    std::vector<uint8_t> source(checked_mul(slab_rows, row_size_src));
    std::vector<uint8_t> encoded(checked_mul(slab_rows, row_size_max));
    std::vector<float> original(checked_mul(slab_rows, cols));
    std::vector<float> decoded(encoded_file ? 0 : original.size());

    std::vector<float> importance;
    const common_imatrix_entry * entry = nullptr;
    double score_divisor = 1.0;
    const auto it = imatrix.entries.find(tensor->name);
    if (it != imatrix.entries.end()) {
        entry = &it->second;
        const size_t matrices = checked_mul(tensor->ne[2], tensor->ne[3]);

        if (entry->sums.size() != checked_mul(cols, matrices) || entry->counts.empty() ||
            (entry->counts.size() != matrices && entry->counts.size() != 1)) {
            throw std::runtime_error(std::string("imatrix shape mismatch for ") + tensor->name);
        }

        size_t observed = 0;
        importance.resize(entry->sums.size());
        for (size_t matrix = 0; matrix < matrices; ++matrix) {
            const int64_t count = entry->counts[entry->counts.size() == 1 ? 0 : matrix];
            if (count < 0) {
                throw std::runtime_error(std::string("negative imatrix count for ") + tensor->name + ", matrix " + std::to_string(matrix));
            }

            observed += count > 0;
            for (size_t col = 0; col < cols; ++col) {
                const size_t j = matrix * cols + col;
                const float sum = entry->sums[j];
                if (!std::isfinite(sum) || sum < 0) {
                    throw std::runtime_error(std::string("invalid imatrix sum for ") + tensor->name + ", matrix " + std::to_string(matrix) + ", column " + std::to_string(col));
                }

                if (count == 0 && sum != 0) {
                    throw std::runtime_error(std::string("nonzero imatrix sum without observations for ") + tensor->name + ", matrix " + std::to_string(matrix));
                }

                // Match llama-quantize's encoding fallback; the raw sums still give these rows zero score weight.
                importance[j] = count > 0 ? sum / (float) count : 1.0f;
            }
        }

        if (observed == 0) {
            throw std::runtime_error(std::string("imatrix has no observations for ") + tensor->name);
        }

        if (observed < matrices) {
            fprintf(stderr, "%s: imatrix covers %zu/%zu matrices; unobserved matrices use uniform encoding weights and zero score weight\n",
                    tensor->name, observed, matrices);
        }

        score_divisor = imatrix.is_legacy ? (double) entry->counts[0] : (double) imatrix.chunk_count * imatrix.chunk_size;
        if (score_divisor <= 0) {
            throw std::runtime_error("imatrix has no calibration token count");
        }
    }

    std::ifstream in(file.path, std::ios::binary);
    in.seekg(plan.offset);
    fprintf(stderr, "[%zu/%zu] %s: %s, %zu candidates, %.3f GiB source\n", tensor_index, n_tensors,
            tensor->name, ggml_type_name(tensor->type), plan.candidates.size(), ggml_nbytes(tensor) / (1024.0 * 1024 * 1024));

    std::vector<double> errors(n_threads), errors_weighted(n_threads);
    std::vector<std::thread> workers;
    tensor_progress progress{tensor->name, rows, n_converts};
    std::atomic<size_t> completed{0};

    for (size_t first = 0; first < rows; first += slab_rows) {
        const size_t nrows = std::min(slab_rows, rows - first);
        const size_t size = nrows * row_size_src;
        in.read((char *) source.data(), size);
        if (!in || !ggml_validate_row_data(tensor->type, source.data(), size)) {
            throw std::runtime_error(std::string("invalid source data for ") + tensor->name);
        }

        to_float(tensor->type, source.data(), original.data(), nrows * cols);

        size_t candidate_index = 0;
        for (auto & c : plan.candidates) {
            if (c.type == tensor->type) {
                continue;
            }

            ++candidate_index;
            completed.store(0, std::memory_order_relaxed);
            auto report_progress = [&](bool force) {
                const size_t done = completed.load(std::memory_order_relaxed);
                const long double work = (long double) first * n_converts + (long double) (candidate_index - 1) * nrows + done;
                progress.report(c.type, candidate_index, first + done, work / ((long double) rows * n_converts), force);
            };

            report_progress(first == 0 && candidate_index == 1);
            ggml_quantize_init(c.type);
            const size_t row_size_dst = ggml_row_size(c.type, cols);

            auto compute = [&](int thread) {
                double error = 0.0;
                double error_weighted = 0.0;
                size_t row = nrows * thread / n_threads;
                const size_t end = nrows * (thread + 1) / n_threads;

                while (row < end) {
                    const size_t matrix = (first + row) / tensor->ne[1];
                    const size_t count = std::min({end - row, (size_t) tensor->ne[1] - (first + row) % tensor->ne[1], size_t(64)});
                    const float * weights = entry ? importance.data() + matrix * cols : nullptr;
                    void * dst = encoded.data() + row * row_size_dst;

                    ggml_quantize_chunk(c.type, original.data() + row * cols, dst, 0, count, cols, weights);
                    if (!encoded_file) {
                        to_float(c.type, dst, decoded.data() + row * cols, count * cols);
                    }

                    for (size_t j = 0; !encoded_file && j < count * cols; ++j) {
                        const double diff = (double) original[row * cols + j] - decoded[row * cols + j];
                        error += diff * diff;
                        if (entry) {
                            error_weighted += diff * diff * entry->sums[matrix * cols + j % cols];
                        }
                    }

                    row += count;
                    completed.fetch_add(count, std::memory_order_relaxed);
                    if (thread == 0) {
                        report_progress(false);
                    }
                }

                errors[thread] = error;
                errors_weighted[thread] = error_weighted / score_divisor;
            };

            try {
                for (int thread = 1; thread < n_threads; ++thread) {
                    workers.emplace_back(compute, thread);
                }
            } catch (...) {
                for (auto & worker : workers) {
                    worker.join();
                }

                throw;
            }

            compute(0);

            for (auto & worker : workers) {
                worker.join();
            }

            workers.clear();
            report_progress(first + nrows == rows && candidate_index == n_converts);

            if (encoded_file && fwrite(encoded.data(), row_size_dst, nrows, encoded_file) != nrows) {
                throw std::runtime_error("cannot write encoded tensor cache");
            }

            for (int thread = 0; thread < n_threads; ++thread) {
                c.error += errors[thread];
                c.error_weighted += errors_weighted[thread];
            }

            if (!std::isfinite(c.error) || !std::isfinite(c.error_weighted)) {
                throw std::runtime_error(std::string("non-finite quantization error for ") + tensor->name);
            }
        }
    }

    return entry != nullptr;
}

struct allocation_option {
    size_t candidate;
    size_t bytes;
    long double error;
};

struct allocation_edge {
    size_t group;
    size_t to;
    size_t bytes;
    long double gain;
};

static std::vector<size_t> allocate_types(const search_params & params, const std::vector<tensor_plan> & plans,
                                          size_t elements, size_t budget, bool report = true) {
    std::vector<std::vector<allocation_option>> groups;
    std::vector<allocation_edge> edges;
    long double error_scale = 0;

    for (const auto & plan : plans) {
        std::vector<allocation_option> options;
        for (size_t j = 0; j < plan.candidates.size(); ++j) {
            const auto & c = plan.candidates[j];
            const long double error = plan.candidates.size() == 1 ? 0 : params.imatrix.empty() ? c.error : c.error_weighted;
            options.push_back({j, plan.counted ? c.size : 0, error});
        }

        std::sort(options.begin(), options.end(), [](const allocation_option & a, const allocation_option & b) {
            if (a.bytes != b.bytes) { return a.bytes < b.bytes; }
            if (a.error != b.error) { return a.error < b.error; }
            return a.candidate < b.candidate;
        });

        // Remove dominated choices, but retain points above the convex hull for discrete search.
        std::vector<allocation_option> frontier;
        for (const auto & option : options) {
            if (frontier.empty() || (option.bytes > frontier.back().bytes && option.error < frontier.back().error)) {
                frontier.push_back(option);
            }
        }

        error_scale += frontier.front().error;
        std::vector<size_t> hull;

        for (size_t j = 0; j < frontier.size(); ++j) {
            while (hull.size() >= 2) {
                const auto & a = frontier[hull[hull.size() - 2]];
                const auto & b = frontier[hull.back()];
                const auto & c = frontier[j];
                if ((a.error - b.error) / (b.bytes - a.bytes) > (b.error - c.error) / (c.bytes - b.bytes)) {
                    break;
                }

                hull.pop_back();
            }

            hull.push_back(j);
        }

        for (size_t j = 1; j < hull.size(); ++j) {
            const auto & a = frontier[hull[j - 1]];
            const auto & b = frontier[hull[j]];
            edges.push_back({groups.size(), hull[j], b.bytes - a.bytes, a.error - b.error});
        }

        groups.push_back(std::move(frontier));
    }

    if (!std::isfinite(error_scale)) {
        throw std::runtime_error("allocation error sum overflow");
    }

    std::sort(edges.begin(), edges.end(), [](const allocation_edge & a, const allocation_edge & b) {
        const long double slope_a = a.gain / a.bytes;
        const long double slope_b = b.gain / b.bytes;
        if (slope_a != slope_b) { return slope_a > slope_b; }
        if (a.group != b.group) { return a.group < b.group; }
        return a.to < b.to;
    });

    struct branch {
        size_t group;
        size_t next;
    };

    std::vector<branch> stack;
    std::vector<size_t> fixed(groups.size(), SIZE_MAX);
    std::vector<size_t> chosen(groups.size());
    std::vector<size_t> best(groups.size());
    long double best_error = std::numeric_limits<long double>::infinity();
    long double root_bound = 0;
    const long double roundoff = 64 * std::numeric_limits<long double>::epsilon() * (groups.size() + edges.size() + 1) * error_scale;
    size_t best_bytes = budget;
    size_t nodes = 0;
    bool pending = true;

    while (pending && nodes < params.max_nodes) {
        ++nodes;
        size_t bytes = 0;
        for (size_t i = 0; i < groups.size(); ++i) {
            chosen[i] = fixed[i] == SIZE_MAX ? 0 : fixed[i];
            bytes = checked_add(bytes, groups[i][chosen[i]].bytes);
        }

        if (bytes <= budget) {
            size_t remaining = budget - bytes;
            size_t fractional = SIZE_MAX;
            long double fractional_gain = 0;

            // Sorted convex-hull upgrades solve the fractional multiple-choice relaxation.
            for (const auto & edge : edges) {
                if (fixed[edge.group] != SIZE_MAX) {
                    continue;
                }

                if (edge.bytes > remaining) {
                    if (remaining > 0) {
                        fractional = edge.group;
                        fractional_gain = edge.gain * ((long double) remaining / edge.bytes);
                    }

                    break;
                }

                chosen[edge.group] = edge.to;
                remaining -= edge.bytes;
            }

            long double error = 0;
            for (size_t i = 0; i < groups.size(); ++i) {
                error += groups[i][chosen[i]].error;
            }

            const long double bound = std::max(0.0L, error - fractional_gain - roundoff);
            if (nodes == 1) {
                root_bound = bound;
            }

            // Round down the fractional group, then try its affordable non-hull choices.
            if (fractional != SIZE_MAX) {
                const auto & options = groups[fractional];
                const auto original = options[chosen[fractional]];
                for (size_t j = chosen[fractional] + 1; j < options.size(); ++j) {
                    if (options[j].bytes - original.bytes > remaining) {
                        break;
                    }

                    chosen[fractional] = j;
                }
            }

            error = 0;
            bytes = 0;
            for (size_t i = 0; i < groups.size(); ++i) {
                error += groups[i][chosen[i]].error;
                bytes = checked_add(bytes, groups[i][chosen[i]].bytes);
            }

            if (error < best_error || (error == best_error && bytes < best_bytes)) {
                best = chosen;
                best_error = error;
                best_bytes = bytes;
            }

            if (fractional != SIZE_MAX && bound < best_error) {
                stack.push_back({fractional, 0});
            }
        }

        pending = false;
        while (!stack.empty()) {
            auto & top = stack.back();
            if (top.next < groups[top.group].size()) {
                fixed[top.group] = top.next++;
                pending = true;
                break;
            }

            fixed[top.group] = SIZE_MAX;
            stack.pop_back();
        }
    }

    const long double lower_bound = pending ? std::min(root_bound, best_error) : best_error;
    if (report) {
        fprintf(stderr, "allocation: %zu/%zu counted payload bytes, %.8Lf bpw\n", best_bytes, budget, 8.0L * best_bytes / elements);
        fprintf(stderr, "%s variable error: %.12Lg, lower bound: %.12Lg, gap: %.12Lg, nodes: %zu (%s)\n",
                params.imatrix.empty() ? "unweighted" : "weighted", best_error, lower_bound, best_error - lower_bound, nodes,
                pending ? "node limit" : "optimal");
    }

    for (size_t i = 0; i < best.size(); ++i) {
        best[i] = groups[i][best[i]].candidate;
    }

    return best;
}

static std::string tensor_pattern(const char * name) {
    std::string pattern = "^";
    for (const unsigned char c : std::string(name)) {
        // Hex escapes also survive llama-quantize's lowercasing of patterns.
        if (c <= ' ' || c >= 127 || (c >= 'A' && c <= 'Z') || c == '=') {
            char escaped[5];
            snprintf(escaped, sizeof(escaped), "\\x%02x", c);
            pattern += escaped;
        } else {
            if (strchr(".^$|()[]{}*+?\\", c)) {
                pattern += '\\';
            }

            pattern += c;
        }
    }

    return pattern + "$";
}

struct eval_file {
    std::string path;
    FILE * file;

    explicit eval_file(const std::string & path) : path(path), file(ggml_fopen(path.c_str(), "w+bx")) {
        if (!file) {
            throw std::runtime_error("cannot create evaluation cache: " + path);
        }
    }

    ~eval_file() {
        fclose(file);
        std::remove(path.c_str());
    }

    eval_file(const eval_file &) = delete;
    eval_file & operator=(const eval_file &) = delete;
};

struct eval_weight {
    size_t file;
    size_t offset;
    size_t size;
};

struct eval_weights {
    std::vector<std::unique_ptr<llama_file>> files;
    std::vector<std::unique_ptr<llama_mmap>> mappings;
    std::vector<ggml_backend_buffer_ptr> buffers;
    std::map<std::string, eval_weight> tensors;

    void map_file(std::unique_ptr<llama_file> file) {
        if (!llama_mmap::SUPPORTED) {
            throw std::runtime_error("global evaluation requires memory mapping support");
        }

        auto mapping = std::make_unique<llama_mmap>(file.get(), 0);
        ggml_backend_buffer_ptr buffer(ggml_backend_cpu_buffer_from_ptr(mapping->addr(), mapping->size()));
        if (!buffer) {
            throw std::runtime_error("cannot wrap mapped CPU weights");
        }

        ggml_backend_buffer_set_usage(buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        files.push_back(std::move(file));
        mappings.push_back(std::move(mapping));
        buffers.push_back(std::move(buffer));
    }

    static void bind(ggml_tensor * tensor, void * user_data) {
        auto & self = *(eval_weights *) user_data;
        const auto it = self.tensors.find(tensor->name);
        if (it == self.tensors.end()) {
            throw std::runtime_error(std::string("no mapped data for ") + tensor->name);
        }

        const auto & weight = it->second;
        const auto & mapping = self.mappings.at(weight.file);
        if (ggml_nbytes(tensor) != weight.size || weight.offset > mapping->size() || weight.size > mapping->size() - weight.offset) {
            throw std::runtime_error(std::string("mapped tensor size mismatch for ") + tensor->name);
        }

        tensor->data = (char *) mapping->addr() + weight.offset;
        tensor->buffer = self.buffers.at(weight.file).get();
    }
};

static gguf_context_ptr eval_metadata(const std::vector<input_file> & files, const std::vector<tensor_plan> & plans, bool reference) {
    gguf_context_ptr metadata(gguf_init_empty());
    gguf_set_kv(metadata.get(), files.front().gguf.get());
    gguf_remove_key(metadata.get(), "split.no");
    gguf_remove_key(metadata.get(), "split.count");
    gguf_remove_key(metadata.get(), "split.tensors.count");

    for (const auto & plan : plans) {
        gguf_add_tensor(metadata.get(), plan.tensor);
        if (!reference) {
            gguf_set_tensor_type(metadata.get(), plan.tensor->name, plan.candidates.front().type);
        }
    }

    return metadata;
}

struct eval_result {
    size_t tokens = 0;
    double nll = 0;
    double kl = 0;
};

static double eval_logsumexp(const float * logits, size_t n) {
    double maximum = -INFINITY;
    for (size_t i = 0; i < n; ++i) {
        if (!std::isfinite(logits[i])) {
            throw std::runtime_error("non-finite evaluation logits");
        }

        maximum = std::max(maximum, (double) logits[i]);
    }

    double sum = 0;
    for (size_t i = 0; i < n; ++i) {
        sum += std::exp((double) logits[i] - maximum);
    }

    return maximum + std::log(sum);
}

struct global_response {
    FILE * baseline;
    size_t rank;
    size_t token = 0;
    double linear = 0;
    std::vector<float> values;
    std::vector<float> logits_base;

    global_response(FILE * baseline, size_t rank, size_t tokens) : baseline(baseline), rank(rank), values(checked_mul(rank, tokens)) {
        rewind(baseline);
    }

    void observe(const float * logits, const std::vector<float> & reference) {
        const size_t n = reference.size();
        logits_base.resize(n);
        if (fread(logits_base.data(), sizeof(float), n, baseline) != n) {
            throw std::runtime_error("cannot read initial recipe logits");
        }

        const double logz_base = eval_logsumexp(logits_base.data(), n);
        const double logz_ref = eval_logsumexp(reference.data(), n);
        double mean = 0;
        for (size_t j = 0; j < n; ++j) {
            const double delta = (double) logits[j] - logits_base[j];
            const double p = std::exp((double) logits_base[j] - logz_base);
            mean += p * delta;
            linear += (p - std::exp((double) reference[j] - logz_ref)) * delta;
        }

        // Two fixed CountSketch maps retain cross-tensor Fisher products without a vocabulary-sized response cache.
        for (size_t j = 0; j < n; ++j) {
            const double centered = ((double) logits[j] - logits_base[j] - mean) * std::exp(0.5 * ((double) logits_base[j] - logz_base)) / std::sqrt(2.0);
            for (size_t k = 0; k < 2; ++k) {
                uint64_t hash = j + (k + 1) * UINT64_C(0x9e3779b97f4a7c15);
                hash = (hash ^ (hash >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
                hash = (hash ^ (hash >> 27)) * UINT64_C(0x94d049bb133111eb);
                hash ^= hash >> 31;
                const size_t bucket = token * rank + k * (rank / 2) + (hash >> 1) % (rank / 2);
                values.at(bucket) += (float) ((hash & 1) ? centered : -centered);
            }
        }

        ++token;
    }

    void normalize() {
        if (!token) {
            throw std::runtime_error("global response has no scored tokens");
        }

        linear /= token;
        const double scale = 1.0 / std::sqrt((double) token);
        for (auto & value : values) {
            value = (float) (value * scale);
            if (!std::isfinite(value)) {
                throw std::runtime_error("non-finite global response");
            }
        }
    }
};

static eval_result eval_tokens(const search_params & params, llama_model * model, const std::vector<llama_token> & tokens, FILE * reference, bool write_reference,
                               FILE * save = nullptr, global_response * response = nullptr) {
    auto cp = llama_context_default_params();
    cp.n_ctx = params.n_ctx;
    cp.n_batch = std::min(params.n_batch, params.n_ctx);
    cp.n_ubatch = cp.n_batch;
    cp.n_threads = params.n_threads;
    cp.n_threads_batch = params.n_threads;
    cp.offload_kqv = false;
    cp.op_offload = false;
    std::unique_ptr<llama_context, decltype(&llama_free)> ctx(llama_init_from_model(model, cp), llama_free);
    if (!ctx) {
        throw std::runtime_error("cannot initialize evaluation context");
    }

    struct eval_batch {
        llama_batch value;
        explicit eval_batch(int n) : value(llama_batch_init(n, 0, 1)) {}
        ~eval_batch() { llama_batch_free(value); }
    } batch(cp.n_batch);

    const size_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    std::vector<float> base(n_vocab);
    eval_result result;
    rewind(reference);
    const size_t count = std::min(tokens.size() - 1, checked_mul(params.n_ctx, params.n_chunks));

    for (size_t chunk = 0; chunk < count; chunk += params.n_ctx) {
        llama_memory_clear(llama_get_memory(ctx.get()), true);
        const size_t end = std::min(count, chunk + params.n_ctx);

        for (size_t first = chunk; first < end; first += cp.n_batch) {
            auto & b = batch.value;
            b.n_tokens = (int32_t) std::min((size_t) cp.n_batch, end - first);
            for (int32_t i = 0; i < b.n_tokens; ++i) {
                b.token[i] = tokens[first + i];
                b.pos[i] = (llama_pos) (first + i - chunk);
                b.n_seq_id[i] = 1;
                b.seq_id[i][0] = 0;
                b.logits[i] = true;
            }

            if (llama_decode(ctx.get(), b) != 0) {
                throw std::runtime_error("evaluation decode failed");
            }

            for (int32_t i = 0; i < b.n_tokens; ++i) {
                const float * logits = llama_get_logits_ith(ctx.get(), i);
                if (!logits) {
                    throw std::runtime_error("missing evaluation logits");
                }

                const double logz = eval_logsumexp(logits, n_vocab);
                result.nll += logz - logits[tokens[first + i + 1]];
                if (save && fwrite(logits, sizeof(float), n_vocab, save) != n_vocab) {
                    throw std::runtime_error("cannot save initial recipe logits");
                }

                if (write_reference) {
                    if (fwrite(logits, sizeof(float), n_vocab, reference) != n_vocab) {
                        throw std::runtime_error("cannot write reference logits");
                    }
                } else {
                    if (fread(base.data(), sizeof(float), n_vocab, reference) != n_vocab) {
                        throw std::runtime_error("cannot read reference logits");
                    }

                    const double logz_base = eval_logsumexp(base.data(), n_vocab);
                    for (size_t j = 0; j < n_vocab; ++j) {
                        const double logp = (double) base[j] - logz_base;
                        result.kl += std::exp(logp) * (logp - ((double) logits[j] - logz));
                    }

                    if (response) {
                        response->observe(logits, base);
                    }
                }

                ++result.tokens;
            }

            fprintf(stderr, "evaluation: %zu/%zu tokens\n", result.tokens, count);
        }
    }

    if (write_reference && fflush(reference) != 0) {
        throw std::runtime_error("cannot flush reference logits");
    }

    if (save && fflush(save) != 0) {
        throw std::runtime_error("cannot flush initial recipe logits");
    }

    return result;
}

using eval_model_ptr = std::unique_ptr<llama_model, decltype(&llama_model_free)>;

static std::vector<tensor_plan> eval_read_recipe(const search_params & params, const std::vector<input_file> & files, const common_imatrix & imatrix, const std::string & path) {
    auto options = params;
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("cannot read recipe: " + path);
    }

    std::string rule;
    while (input >> rule) {
        const size_t sep = rule.find('=');
        if (sep == std::string::npos || sep == 0) {
            throw std::runtime_error("invalid recipe rule: " + rule);
        }

        auto types = parse_types(rule.substr(sep + 1));
        if (types.size() != 1) {
            throw std::runtime_error("evaluation recipes require one type per rule");
        }

        options.rules.push_back({std::regex(rule.substr(0, sep)), std::move(types)});
    }

    auto plans = plan_tensors(options, files, imatrix);
    for (const auto & plan : plans) {
        if (plan.candidates.size() != 1) {
            throw std::runtime_error(std::string("ambiguous evaluation type for ") + plan.tensor->name);
        }
    }

    return plans;
}

struct eval_cache {
    const search_params & params;
    const std::vector<input_file> & files;
    const common_imatrix & imatrix;
    eval_file encoded;
    eval_weights weights;
    std::map<std::pair<std::string, ggml_type>, eval_weight> cache;
    size_t cache_size = 0;

    eval_cache(const search_params & params, const std::vector<input_file> & files, const common_imatrix & imatrix)
        : params(params), files(files), imatrix(imatrix), encoded(params.eval_cache + ".weights") {
        for (const auto & file : files) {
            weights.map_file(std::make_unique<llama_file>(file.path.c_str(), "rb"));
        }
    }

    eval_model_ptr load(std::vector<tensor_plan> & plans, bool reference = false) {
        if (weights.files.size() > files.size()) {
            weights.buffers.pop_back();
            weights.mappings.pop_back();
            weights.files.pop_back();
        }

        llama_file cache_file(encoded.file);
        cache_file.seek(cache_size, SEEK_SET);
        for (size_t i = 0; i < plans.size(); ++i) {
            auto & plan = plans[i];
            const auto & candidate = plan.candidates.front();
            if (reference || candidate.type == plan.tensor->type) {
                weights.tensors[plan.tensor->name] = {plan.file, plan.offset, ggml_nbytes(plan.tensor)};
                continue;
            }

            const auto key = std::make_pair(std::string(plan.tensor->name), candidate.type);
            auto it = cache.find(key);
            if (it == cache.end()) {
                const size_t padding = (64 - cache_size % 64) % 64;
                const uint8_t zeros[64] = {};
                if (fwrite(zeros, 1, padding, encoded.file) != padding) {
                    throw std::runtime_error("cannot pad encoded tensor cache");
                }

                cache_size = checked_add(cache_size, padding);
                const eval_weight weight{files.size(), cache_size, candidate.size};
                auto encoding_params = params;
                encoding_params.target_bpw = 0;
                measure_tensor(encoding_params, files[plan.file], plan, imatrix, i + 1, plans.size(), encoded.file);
                cache_size = checked_add(cache_size, candidate.size);
                it = cache.emplace(key, weight).first;
            }

            weights.tensors[plan.tensor->name] = it->second;
        }

        if (fflush(encoded.file) != 0) {
            throw std::runtime_error("cannot flush encoded tensor cache");
        }

        if (cache_size) {
            weights.map_file(std::make_unique<llama_file>(encoded.file));
        }

        auto metadata = eval_metadata(files, plans, reference);
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(llama_model_init_from_buffers(metadata.get(), eval_weights::bind, &weights), llama_model_free);
        if (!model) {
            throw std::runtime_error("cannot load recipe model for global evaluation");
        }

        return model;
    }
};

static std::vector<llama_token> eval_calibration_tokens(const search_params & params, llama_model * model) {
    std::ifstream input(params.calibration_file, std::ios::binary);
    if (!input) {
        throw std::runtime_error("cannot read calibration text");
    }

    const std::string text((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    auto tokens = common_tokenize(llama_model_get_vocab(model), text, true);
    if (tokens.size() < 2) {
        throw std::runtime_error("calibration text requires at least two tokens");
    }

    tokens.resize(std::min(tokens.size(), checked_add(checked_mul(params.n_ctx, params.n_chunks), 1)));
    return tokens;
}

static std::pair<size_t, size_t> eval_payload(const std::vector<tensor_plan> & plans, bool reference = false) {
    size_t elements = 0;
    size_t bytes = 0;
    for (const auto & plan : plans) {
        if (plan.counted) {
            elements = checked_add(elements, ggml_nelements(plan.tensor));
            bytes = checked_add(bytes, reference ? ggml_nbytes(plan.tensor) : plan.candidates.front().size);
        }
    }

    return {elements, bytes};
}

static void evaluate_recipes(const search_params & params, const std::vector<input_file> & files, const common_imatrix & imatrix) {
    std::vector<std::vector<tensor_plan>> recipes;
    for (const auto & path : params.eval_recipes) {
        recipes.push_back(eval_read_recipe(params, files, imatrix, path));
    }

    ggml_backend_load_all();
    llama_backend_init();
    eval_file logits(params.eval_cache + ".logits");
    eval_cache cache(params, files, imatrix);
    std::vector<llama_token> tokens;
    eval_result base_result;
    {
        auto model = cache.load(recipes.front(), true);
        tokens = eval_calibration_tokens(params, model.get());
        base_result = eval_tokens(params, model.get(), tokens, logits.file, true);
    }

    auto report = [](const char * name, std::pair<size_t, size_t> payload, const eval_result & result) {
        const auto [elements, bytes] = payload;
        printf("%s\t%zu\t%zu\t%.8f\t%zu\t%.12g\t%.12g\n", name, elements, bytes,
               elements ? 8.0 * (double) bytes / (double) elements : 0.0, result.tokens,
               std::exp(result.nll / result.tokens), std::max(0.0, result.kl / result.tokens));
    };

    printf("recipe\telements\tbytes\tbpw\ttokens\tppl\tkl\n");
    report("reference", eval_payload(recipes.front(), true), base_result);
    for (size_t r = 0; r < recipes.size(); ++r) {
        auto model = cache.load(recipes[r]);
        report(params.eval_recipes[r].c_str(), eval_payload(recipes[r]), eval_tokens(params, model.get(), tokens, logits.file, false));
    }
}

struct global_option {
    size_t offset = SIZE_MAX;
    double linear = 0;
};

static double global_dot(const std::vector<float> & a, const std::vector<float> & b) {
    double result = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        result += (double) a[i] * b[i];
    }

    return result;
}

static std::vector<size_t> global_solve(const search_params & params, std::vector<tensor_plan> plans,
                                      const std::vector<std::vector<global_option>> & options, FILE * responses,
                                      size_t dimensions, size_t elements, size_t budget, const std::vector<size_t> & initial) {
    llama_file response_file(responses);
    const size_t row_bytes = checked_mul(dimensions, sizeof(float));
    std::vector<float> current(dimensions), vertex(dimensions), work(dimensions);
    auto read = [&](const global_option & option) {
        if (option.offset == SIZE_MAX) {
            std::fill(work.begin(), work.end(), 0);
            return;
        }

        if (option.offset > response_file.size() || row_bytes > response_file.size() - option.offset) {
            throw std::runtime_error("global response outside cache");
        }

        response_file.seek(option.offset, SEEK_SET);
        response_file.read_raw(work.data(), row_bytes);
    };

    double linear = 0;
    double best_value = 0;
    auto best = initial;
    for (size_t iteration = 0; iteration < 64; ++iteration) {
        for (size_t i = 0; i < plans.size(); ++i) {
            double minimum = INFINITY;
            for (size_t j = 0; j < options[i].size(); ++j) {
                read(options[i][j]);
                const double gradient = options[i][j].linear + global_dot(current, work);
                plans[i].candidates[j].error = gradient;
                minimum = std::min(minimum, gradient);
            }

            // Per-group constants leave the linear oracle unchanged and keep its error bounds nonnegative.
            for (auto & candidate : plans[i].candidates) {
                candidate.error -= minimum;
                candidate.error_weighted = candidate.error;
            }
        }

        const auto chosen = allocate_types(params, plans, elements, budget, false);
        std::fill(vertex.begin(), vertex.end(), 0);
        double vertex_linear = 0;
        for (size_t i = 0; i < plans.size(); ++i) {
            const auto & option = options[i][chosen[i]];
            vertex_linear += option.linear;
            read(option);
            for (size_t k = 0; k < dimensions; ++k) {
                vertex[k] += work[k];
            }
        }

        const double value = vertex_linear + 0.5 * global_dot(vertex, vertex);
        if (value < best_value) {
            best_value = value;
            best = chosen;
        }

        double derivative = vertex_linear - linear;
        double curvature = 0;
        for (size_t k = 0; k < dimensions; ++k) {
            const double delta = (double) vertex[k] - current[k];
            derivative += current[k] * delta;
            curvature += delta * delta;
        }

        if (derivative >= -1e-12) {
            break;
        }

        const double step = curvature > 0 ? std::min(1.0, -derivative / curvature) : 1.0;
        linear += step * (vertex_linear - linear);
        for (size_t k = 0; k < dimensions; ++k) {
            current[k] = (float) (current[k] + step * ((double) vertex[k] - current[k]));
        }

        fprintf(stderr, "global quadratic: iteration %zu, predicted best KL change %.12g\n", iteration + 1, best_value);
    }

    return best;
}

static void global_search(const search_params & params, const std::vector<input_file> & files, const common_imatrix & imatrix) {
    auto plans = plan_tensors(params, files, imatrix);
    auto initial_plans = eval_read_recipe(params, files, imatrix, params.global_recipe);
    const auto [elements, initial_bytes] = eval_payload(initial_plans);
    if (!elements) {
        throw std::runtime_error("global search requires counted elements");
    }

    const long double target = std::floor(params.target_bpw * (long double) elements / 8);
    const size_t budget = target >= (long double) SIZE_MAX ? SIZE_MAX : (size_t) target;
    if (initial_bytes > budget) {
        throw std::runtime_error("initial recipe exceeds the BPW budget");
    }

    std::vector<size_t> initial;
    std::vector<std::vector<global_option>> options;
    size_t probes = 0;
    for (size_t i = 0; i < plans.size(); ++i) {
        if (!params.global_tensors.empty() && !matches(params.global_tensors, plans[i].tensor->name)) {
            plans[i].candidates = initial_plans[i].candidates;
        }

        const auto & candidates = plans[i].candidates;
        const auto type = initial_plans[i].candidates.front().type;
        const auto it = std::find_if(candidates.begin(), candidates.end(), [&](const candidate & c) { return c.type == type; });
        if (it == candidates.end()) {
            throw std::runtime_error(std::string("initial type is not an allowed candidate for ") + plans[i].tensor->name);
        }

        initial.push_back(it - candidates.begin());
        options.emplace_back(candidates.size());
        probes = checked_add(probes, candidates.size() - 1);
    }

    fprintf(stderr, "global search: %zu alternatives, one full calibration evaluation per alternative\n", probes);
    ggml_backend_load_all();
    llama_backend_init();
    eval_file reference(params.eval_cache + ".logits");
    eval_file baseline(params.eval_cache + ".baseline");
    eval_file responses(params.eval_cache + ".responses");
    eval_cache cache(params, files, imatrix);
    std::vector<llama_token> tokens;
    {
        auto model = cache.load(initial_plans, true);
        tokens = eval_calibration_tokens(params, model.get());
        const size_t dimensions = checked_mul(tokens.size() - 1, params.response_rank);
        if (checked_mul(checked_mul(dimensions, sizeof(float)), 4) > params.max_buffer_size) {
            throw std::runtime_error("global response buffers exceed --max-buffer-size; reduce --chunks, --ctx-size, or --response-rank");
        }

        eval_tokens(params, model.get(), tokens, reference.file, true);
    }

    eval_result initial_result;
    {
        auto model = cache.load(initial_plans);
        initial_result = eval_tokens(params, model.get(), tokens, reference.file, false, baseline.file);
    }

    fprintf(stderr, "global initial: KL %.12g, PPL %.12g, %zu/%zu counted payload bytes\n",
            initial_result.kl / initial_result.tokens, std::exp(initial_result.nll / initial_result.tokens), initial_bytes, budget);

    const size_t dimensions = checked_mul(initial_result.tokens, params.response_rank);
    const size_t row_bytes = checked_mul(dimensions, sizeof(float));
    size_t response_bytes = 0;
    size_t probe = 0;
    for (size_t i = 0; i < plans.size(); ++i) {
        for (size_t j = 0; j < plans[i].candidates.size(); ++j) {
            if (j == initial[i]) {
                continue;
            }

            fprintf(stderr, "global response [%zu/%zu]: %s=%s\n", ++probe, probes, plans[i].tensor->name, ggml_type_name(plans[i].candidates[j].type));
            auto trial = initial_plans;
            trial[i].candidates = {plans[i].candidates[j]};
            global_response response(baseline.file, params.response_rank, initial_result.tokens);
            {
                auto model = cache.load(trial);
                eval_tokens(params, model.get(), tokens, reference.file, false, nullptr, &response);
            }

            response.normalize();
            options[i][j] = {response_bytes, response.linear};
            if (fwrite(response.values.data(), sizeof(float), dimensions, responses.file) != dimensions) {
                throw std::runtime_error("cannot write global response cache");
            }

            response_bytes = checked_add(response_bytes, row_bytes);
        }
    }

    if (fflush(responses.file) != 0) {
        throw std::runtime_error("cannot flush global response cache");
    }

    auto selected = probes ? global_solve(params, plans, options, responses.file, dimensions, elements, budget, initial) : initial;
    if (selected != initial) {
        auto trial = initial_plans;
        for (size_t i = 0; i < trial.size(); ++i) {
            trial[i].candidates = {plans[i].candidates[selected[i]]};
        }

        eval_result actual;
        {
            auto model = cache.load(trial);
            actual = eval_tokens(params, model.get(), tokens, reference.file, false);
        }

        const bool improved = actual.kl + 1e-12 * actual.tokens < initial_result.kl;
        fprintf(stderr, "global proposal: KL %.12g, PPL %.12g, %s\n", actual.kl / actual.tokens,
                std::exp(actual.nll / actual.tokens), improved ? "accepted" : "rejected; retaining initial recipe");
        if (!improved) {
            selected = initial;
        }
    } else {
        fprintf(stderr, "global search: no improving quadratic proposal; retaining initial recipe\n");
    }

    size_t selected_bytes = 0;
    for (size_t i = 0; i < plans.size(); ++i) {
        const auto & choice = plans[i].candidates[selected[i]];
        if (plans[i].counted) {
            selected_bytes = checked_add(selected_bytes, choice.size);
        }

        printf("%s=%s\n", tensor_pattern(plans[i].tensor->name).c_str(), ggml_type_name(choice.type));
    }

    fprintf(stderr, "global allocation: %zu/%zu counted payload bytes, %.8Lf bpw\n", selected_bytes, budget, 8.0L * selected_bytes / elements);
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    ggml_time_init();

    if (argc == 2 && (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0)) {
        usage(argv[0]);
        return 0;
    }

    int result = 0;
    try {
        const auto params = parse_params(argc, argv);

        if (!params.eval_recipes.empty() || !params.global_recipe.empty()) {
            llama_log_set([](ggml_log_level level, const char * message, void *) {
                if (level == GGML_LOG_LEVEL_WARN || level == GGML_LOG_LEVEL_ERROR) {
                    fputs(message, stderr);
                }
            }, nullptr);
        }

        common_imatrix imatrix;
        if (!params.imatrix.empty() && !common_imatrix_load(params.imatrix, imatrix)) {
            throw std::runtime_error("cannot load imatrix");
        }

        const auto files = open_inputs(params.model);
        if (!params.global_recipe.empty()) {
            global_search(params, files, imatrix);
            if (fflush(stdout) != 0 || ferror(stdout)) {
                throw std::runtime_error("cannot write global search recipe");
            }

            llama_backend_free();
            ggml_quantize_free();
            return 0;
        }

        if (!params.eval_recipes.empty()) {
            evaluate_recipes(params, files, imatrix);
            if (fflush(stdout) != 0 || ferror(stdout)) {
                throw std::runtime_error("cannot write evaluation results");
            }

            llama_backend_free();
            ggml_quantize_free();
            return 0;
        }

        auto plans = plan_tensors(params, files, imatrix);

        if (params.target_bpw > 0 && !params.imatrix.empty()) {
            for (const auto & plan : plans) {
                if (plan.candidates.size() > 1 && imatrix.entries.count(plan.tensor->name) == 0) {
                    throw std::runtime_error(std::string("missing imatrix for searchable tensor: ") + plan.tensor->name + "; fix its type with --tensor-type or use --exclude-tensors");
                }
            }
        }

        size_t elements = 0;
        size_t size_min = 0;
        size_t size_max = 0;
        for (const auto & plan : plans) {
            size_t bytes_min = SIZE_MAX;
            size_t bytes_max = 0;
            for (const auto & c : plan.candidates) {
                bytes_min = std::min(bytes_min, c.size);
                bytes_max = std::max(bytes_max, c.size);
            }

            if (plan.counted) {
                elements = checked_add(elements, ggml_nelements(plan.tensor));
                size_min = checked_add(size_min, bytes_min);
                size_max = checked_add(size_max, bytes_max);
            }
        }

        size_t budget = 0;
        if (params.target_bpw > 0) {
            if (elements == 0) {
                throw std::runtime_error("BPW allocation requires counted elements");
            }

            const long double target_bytes = std::floor(params.target_bpw * (long double) elements / 8);
            budget = target_bytes >= (long double) SIZE_MAX ? SIZE_MAX : (size_t) target_bytes;
            if (size_min > budget) {
                throw std::runtime_error("infeasible BPW limit: minimum is " + std::to_string(size_min) + " counted payload bytes");
            }
        }

        if (params.target_bpw == 0) {
            printf("tensor\ttype\telements\tbytes\tbpw\tsearch\tcount\terror\tweighted_error\n");
        }

        for (size_t i = 0; i < plans.size(); ++i) {
            auto & plan = plans[i];
            const bool weighted = measure_tensor(params, files[plan.file], plan, imatrix, i + 1, plans.size());
            const size_t nelements = ggml_nelements(plan.tensor);

            for (const auto & c : plan.candidates) {
                if (params.target_bpw > 0) {
                    continue;
                }

                printf("%s\t%s\t%zu\t%zu\t%.8f\t%d\t%d\t", plan.tensor->name, ggml_type_name(c.type),
                       nelements, c.size, nelements ? 8.0 * (double) c.size / (double) nelements : 0.0, plan.candidates.size() > 1, plan.counted);

                if (params.dry_run && c.type != plan.tensor->type) {
                    printf("-\t-\n");
                } else {
                    printf("%.17g\t", c.error);
                    if (weighted) {
                        printf("%.17g\n", c.error_weighted);
                    } else {
                        printf("-\n");
                    }
                }
            }
        }

        fprintf(stderr, "counted: %zu elements, %zu..%zu payload bytes", elements, size_min, size_max);
        if (elements) {
            fprintf(stderr, ", %.8f..%.8f bpw", 8.0 * (double) size_min / (double) elements, 8.0 * (double) size_max / (double) elements);
        }
        fprintf(stderr, "\n");

        if (params.target_bpw > 0) {
            const auto selected = allocate_types(params, plans, elements, budget);
            for (size_t i = 0; i < plans.size(); ++i) {
                const auto & plan = plans[i];
                printf("%s=%s\n", tensor_pattern(plan.tensor->name).c_str(), ggml_type_name(plan.candidates[selected[i]].type));
            }
        }

        if (fflush(stdout) != 0 || ferror(stdout)) {
            throw std::runtime_error("cannot write results");
        }
    } catch (const std::exception & error) {
        fprintf(stderr, "error: %s\n", error.what());
        result = 1;
    }

    ggml_quantize_free();
    return result;
}
