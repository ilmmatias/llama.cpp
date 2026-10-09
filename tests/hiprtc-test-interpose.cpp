#include "hiprtc-test-interpose.h"
#include <chrono>
#include <hip/hip_runtime_api.h>
#include <hip/hiprtc.h>
#include <dlfcn.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

static std::mutex observer_mutex;
static uint64_t counters[HIPRTC_TEST_COUNTER_COUNT] = {};
static unsigned captures = 0;

static void observe(hiprtc_test_counter counter, bool setup = false) {
    std::lock_guard<std::mutex> lock(observer_mutex);
    ++counters[counter];
    if (setup && captures) {
        ++counters[HIPRTC_TEST_SETUP_DURING_CAPTURE];
    }
}

template <typename T>
static T real_api(const char * name) {
    auto function = reinterpret_cast<T>(dlsym(RTLD_NEXT, name));
    if (!function) {
        std::fprintf(stderr, "HIP observer: %s: %s\n", name, dlerror());
        std::abort();
    }

    return function;
}

extern "C" void hiprtc_test_snapshot(uint64_t * destination) {
    std::lock_guard<std::mutex> lock(observer_mutex);
    std::memcpy(destination, counters, sizeof(counters));
}

extern "C" hiprtcResult hiprtcCreateProgram(hiprtcProgram * program, const char * source, const char * name,
                                          int count, const char * const * headers, const char * const * names) {
    static auto call = real_api<decltype(&hiprtcCreateProgram)>("hiprtcCreateProgram");
    observe(HIPRTC_TEST_CREATE, true);

    const char * fail = std::getenv("GGML_TEST_HIPRTC_FAIL_CREATE");
    if (fail && std::atoi(fail)) {
        source = "__global__ void broken( {";
    }

    return call(program, source, name, count, headers, names);
}

extern "C" hiprtcResult hiprtcCompileProgram(hiprtcProgram program, int count, const char * const * options) {
    static auto call = real_api<decltype(&hiprtcCompileProgram)>("hiprtcCompileProgram");
    observe(HIPRTC_TEST_COMPILE, true);
    return call(program, count, options);
}

extern "C" hiprtcResult hiprtcVersion(int * major, int * minor) {
    static auto call = real_api<decltype(&hiprtcVersion)>("hiprtcVersion");
    observe(HIPRTC_TEST_VERSION, true);
    return call(major, minor);
}

extern "C" hipError_t hipModuleLoadData(hipModule_t * module, const void * image) {
    static auto call = real_api<decltype(&hipModuleLoadData)>("hipModuleLoadData");
    observe(HIPRTC_TEST_LOAD, true);

    const auto start = std::chrono::steady_clock::now();
    const auto status = call(module, image);
    const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count();

    {
        std::lock_guard<std::mutex> lock(observer_mutex);
        counters[HIPRTC_TEST_LOAD_NS] += elapsed;
    }

    return status;
}

extern "C" hipError_t hipModuleUnload(hipModule_t module) {
    static auto call = real_api<decltype(&hipModuleUnload)>("hipModuleUnload");
    observe(HIPRTC_TEST_UNLOAD, true);
    return call(module);
}

extern "C" hipError_t hipModuleLaunchKernel(hipFunction_t function, unsigned gx, unsigned gy, unsigned gz,
                                          unsigned bx, unsigned by, unsigned bz, unsigned shared,
                                          hipStream_t stream, void ** arguments, void ** extra) {
    static auto call = real_api<decltype(&hipModuleLaunchKernel)>("hipModuleLaunchKernel");
    observe(HIPRTC_TEST_MODULE_LAUNCH);
    return call(function, gx, gy, gz, bx, by, bz, shared, stream, arguments, extra);
}

extern "C" hipError_t hipFuncGetAttribute(int * value, hipFunction_attribute attribute, hipFunction_t function) {
    static auto call = real_api<decltype(&hipFuncGetAttribute)>("hipFuncGetAttribute");
    observe(HIPRTC_TEST_RESOURCE_QUERY, true);
    return call(value, attribute, function);
}

extern "C" hipError_t hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(int * blocks, hipFunction_t function,
                                                                     int threads, size_t shared) {
    static auto call = real_api<decltype(&hipModuleOccupancyMaxActiveBlocksPerMultiprocessor)>("hipModuleOccupancyMaxActiveBlocksPerMultiprocessor");
    observe(HIPRTC_TEST_RESOURCE_QUERY, true);
    return call(blocks, function, threads, shared);
}

extern "C" hipError_t hipMalloc(void ** pointer, size_t size) {
    static auto call = real_api<hipError_t (*)(void **, size_t)>("hipMalloc");
    observe(HIPRTC_TEST_ALLOCATE, true);
    return call(pointer, size);
}

extern "C" hipError_t hipStreamBeginCapture(hipStream_t stream, hipStreamCaptureMode mode) {
    static auto call = real_api<decltype(&hipStreamBeginCapture)>("hipStreamBeginCapture");
    {
        std::lock_guard<std::mutex> lock(observer_mutex);
        ++counters[HIPRTC_TEST_CAPTURE_BEGIN];
        ++captures;
    }

    const auto status = call(stream, mode);
    if (status != hipSuccess) {
        std::lock_guard<std::mutex> lock(observer_mutex);
        --captures;
    }

    return status;
}

extern "C" hipError_t hipStreamEndCapture(hipStream_t stream, hipGraph_t * graph) {
    static auto call = real_api<decltype(&hipStreamEndCapture)>("hipStreamEndCapture");
    const auto status = call(stream, graph);

    {
        std::lock_guard<std::mutex> lock(observer_mutex);
        ++counters[HIPRTC_TEST_CAPTURE_END];
        if (!captures) {
            std::abort();
        }

        --captures;
    }

    return status;
}

extern "C" hipError_t hipGraphLaunch(hipGraphExec_t graph, hipStream_t stream) {
    static auto call = real_api<decltype(&hipGraphLaunch)>("hipGraphLaunch");
    observe(HIPRTC_TEST_GRAPH_LAUNCH);
    return call(graph, stream);
}

extern "C" hipError_t hipEventRecord(hipEvent_t event, hipStream_t stream) {
    static auto call = real_api<decltype(&hipEventRecord)>("hipEventRecord");
    observe(HIPRTC_TEST_EVENT_RECORD);
    return call(event, stream);
}

extern "C" hipError_t hipStreamWaitEvent(hipStream_t stream, hipEvent_t event, unsigned flags) {
    static auto call = real_api<decltype(&hipStreamWaitEvent)>("hipStreamWaitEvent");
    observe(HIPRTC_TEST_STREAM_WAIT);
    return call(stream, event, flags);
}
