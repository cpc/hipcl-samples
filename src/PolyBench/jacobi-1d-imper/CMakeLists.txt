add_hipcl_test(jacobi1D jacobi1D Passed jacobi1D.cu)
target_include_directories(jacobi1D PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../utilities>)

