add_hipcl_test(bicg bicg Passed bicg.cu)
target_include_directories(bicg PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../utilities>)

