#include "common.h"
#include "imatrix-loader.h"
#include "llama.h"
#include "ggml-cpp.h"

#include "../../src/llama-ext.h"

#include <algorithm>
#include <cctype>
#include <clocale>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
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
                   arg == "--max-nodes" || arg == "-t" || arg == "--threads") {
            if (++i == argc) {
                throw std::runtime_error("missing value for " + arg);
            }

            const std::string value = argv[i];
            if (arg == "--imatrix") {
                params.imatrix = value;
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

static bool measure_tensor(const search_params & params, const input_file & file, tensor_plan & plan,
                           const common_imatrix & imatrix) {
    if (params.dry_run || (params.target_bpw > 0 && plan.candidates.size() == 1)) {
        return false;
    }

    const auto * tensor = plan.tensor;
    bool converts = false;
    size_t row_size_max = 0;
    for (const auto & c : plan.candidates) {
        converts |= c.type != tensor->type;
        row_size_max = std::max(row_size_max, ggml_row_size(c.type, tensor->ne[0]));
    }

    if (!converts) {
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
    std::vector<float> decoded(original.size());

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
    fprintf(stderr, "%s: %s, %zu candidates\n", tensor->name, ggml_type_name(tensor->type), plan.candidates.size());

    std::vector<double> errors(n_threads), errors_weighted(n_threads);
    std::vector<std::thread> workers;

    for (size_t first = 0; first < rows; first += slab_rows) {
        const size_t nrows = std::min(slab_rows, rows - first);
        const size_t size = nrows * row_size_src;
        in.read((char *) source.data(), size);
        if (!in || !ggml_validate_row_data(tensor->type, source.data(), size)) {
            throw std::runtime_error(std::string("invalid source data for ") + tensor->name);
        }

        to_float(tensor->type, source.data(), original.data(), nrows * cols);

        for (auto & c : plan.candidates) {
            if (c.type == tensor->type) {
                continue;
            }

            ggml_quantize_init(c.type);
            const size_t row_size_dst = ggml_row_size(c.type, cols);

            auto compute = [&](int thread) {
                double error = 0.0;
                double error_weighted = 0.0;
                size_t row = nrows * thread / n_threads;
                const size_t end = nrows * (thread + 1) / n_threads;

                while (row < end) {
                    const size_t matrix = (first + row) / tensor->ne[1];
                    const size_t count = std::min(end - row, (size_t) tensor->ne[1] - (first + row) % tensor->ne[1]);
                    const float * weights = entry ? importance.data() + matrix * cols : nullptr;
                    void * dst = encoded.data() + row * row_size_dst;

                    ggml_quantize_chunk(c.type, original.data() + row * cols, dst, 0, count, cols, weights);
                    to_float(c.type, dst, decoded.data() + row * cols, count * cols);

                    for (size_t j = 0; j < count * cols; ++j) {
                        const double diff = (double) original[row * cols + j] - decoded[row * cols + j];
                        error += diff * diff;
                        if (entry) {
                            error_weighted += diff * diff * entry->sums[matrix * cols + j % cols];
                        }
                    }

                    row += count;
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
                                          size_t elements, size_t budget) {
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
    fprintf(stderr, "allocation: %zu/%zu counted payload bytes, %.8Lf bpw\n", best_bytes, budget, 8.0L * best_bytes / elements);
    fprintf(stderr, "%s variable error: %.12Lg, lower bound: %.12Lg, gap: %.12Lg, nodes: %zu (%s)\n",
            params.imatrix.empty() ? "unweighted" : "weighted", best_error, lower_bound, best_error - lower_bound, nodes,
            pending ? "node limit" : "optimal");

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

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    if (argc == 2 && (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0)) {
        usage(argv[0]);
        return 0;
    }

    int result = 0;
    try {
        const auto params = parse_params(argc, argv);

        common_imatrix imatrix;
        if (!params.imatrix.empty() && !common_imatrix_load(params.imatrix, imatrix)) {
            throw std::runtime_error("cannot load imatrix");
        }

        const auto files = open_inputs(params.model);
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

        for (auto & plan : plans) {
            const bool weighted = measure_tensor(params, files[plan.file], plan, imatrix);
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
