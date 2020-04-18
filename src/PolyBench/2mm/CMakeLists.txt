add_hipcl_test(2mm 2mm Passed 2mm.cu)
target_include_directories(2mm PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../utilities>)

