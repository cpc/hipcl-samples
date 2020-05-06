add_hipcl_test(lu lu Passed lu.cu)
target_include_directories(lu PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../utilities>)

