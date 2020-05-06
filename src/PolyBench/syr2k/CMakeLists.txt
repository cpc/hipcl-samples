add_hipcl_test(syr2k syr2k Passed syr2k.cu)
target_include_directories(syr2k PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../utilities>)

