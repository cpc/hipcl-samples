add_hipcl_test(correlation correlation Passed correlation.cu)
target_include_directories(correlation PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../utilities>)

