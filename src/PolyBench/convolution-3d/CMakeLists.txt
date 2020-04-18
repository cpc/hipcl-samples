add_hipcl_test(3DConvolution 3DConvolution Passed 3DConvolution.cu)
target_include_directories(3DConvolution PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../utilities>)

