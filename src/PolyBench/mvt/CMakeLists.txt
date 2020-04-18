add_hipcl_test(mvt mvt Passed mvt.cu)
target_include_directories(mvt PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../utilities>)

