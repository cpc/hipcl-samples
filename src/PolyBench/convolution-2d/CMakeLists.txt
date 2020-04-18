add_hipcl_test(2DConvolution 2DConvolution Passed 2DConvolution.cu)
target_include_directories(2DConvolution PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../utilities>)

