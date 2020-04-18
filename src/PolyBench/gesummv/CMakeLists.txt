add_hipcl_test(gesummv gesummv Passed gesummv.cu)
target_include_directories(gesummv PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../utilities>)

