add_hipcl_test(gemm gemm Passed gemm.cu)
target_include_directories(gemm PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../utilities>)

