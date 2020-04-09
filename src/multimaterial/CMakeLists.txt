message(STATUS "downloading volfrac.dat.tgz")
file(DOWNLOAD https://github.com/lanl/MultiMatTest/raw/master/volfrac.dat.tgz "${CMAKE_CURRENT_BINARY_DIR}/volfrac.dat.tgz")

message(STATUS "extracting volfrac.dat.tgz")
execute_process(COMMAND ${CMAKE_COMMAND} -E tar xzf "${CMAKE_CURRENT_BINARY_DIR}/volfrac.dat.tgz"
                WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")

add_hipcl_binary(multimat_hipcl compact.hip.cpp full_matrix.cpp multimat.cpp)

add_hipcl_binary(multimat_hipcl_F compact.hip.cpp full_matrix.cpp multimat.cpp)

add_hipcl_binary(multimat_hipcl_FL compact.hip.cpp full_matrix.cpp multimat.cpp)

target_compile_definitions(multimat_hipcl PRIVATE VERIFY)

target_compile_definitions(multimat_hipcl_F PRIVATE VERIFY FUSED)

target_compile_definitions(multimat_hipcl_FL PRIVATE VERIFY FUSED LINKED)
