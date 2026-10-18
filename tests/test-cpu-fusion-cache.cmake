function(run_cache directory mode)
    execute_process(
        COMMAND "${CMAKE_COMMAND}" -E env "GGML_CPU_FUSION_CACHE_DIR=${directory}"
            "${TEST_EXECUTABLE}" cache "${mode}"
        RESULT_VARIABLE result
        OUTPUT_VARIABLE output
        ERROR_VARIABLE error)
    if (NOT result EQUAL 0)
        message(FATAL_ERROR "CPU fusion cache ${mode}: ${result}\n${output}\n${error}")
    endif()
endfunction()

file(REMOVE_RECURSE "${CACHE_DIR}")
run_cache("${CACHE_DIR}" cold)
file(GLOB objects "${CACHE_DIR}/*.o")
list(LENGTH objects count)
if (NOT count EQUAL 2)
    message(FATAL_ERROR "Expected two cached CPU fusion objects, found ${count}")
endif()

# Reverse the compile order and change tensor sizes in a fresh process.
run_cache("${CACHE_DIR}" warm)

# Header: 48-byte symbol name, newline, and 32-byte hexadecimal checksum.
set(header_bytes 81)
math(EXPR truncated_bytes "${header_bytes} - 1")
foreach(object IN LISTS objects)
    file(READ "${object}" header LIMIT ${truncated_bytes})
    file(WRITE "${object}" "${header}")
endforeach()
run_cache("${CACHE_DIR}" cold)
run_cache("${CACHE_DIR}" warm)

foreach(object IN LISTS objects)
    file(READ "${object}" header LIMIT ${header_bytes})
    file(WRITE "${object}" "${header}damaged object")
endforeach()
run_cache("${CACHE_DIR}" cold)
run_cache("${CACHE_DIR}" warm)
run_cache(0 cold)

file(WRITE "${CACHE_DIR}/blocked" "")
run_cache("${CACHE_DIR}/blocked/cache" cold)
file(REMOVE_RECURSE "${CACHE_DIR}")
