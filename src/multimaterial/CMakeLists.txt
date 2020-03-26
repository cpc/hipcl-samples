add_hipcl_binary(multimat_hipcl compact.hip.cpp full_matrix.cpp multimat.cpp)

add_hipcl_binary(multimat_hipcl_F compact.hip.cpp full_matrix.cpp multimat.cpp)

add_hipcl_binary(multimat_hipcl_FL compact.hip.cpp full_matrix.cpp multimat.cpp)

target_compile_definitions(multimat_hipcl PRIVATE VERIFY)

target_compile_definitions(multimat_hipcl_F PRIVATE VERIFY FUSED)

target_compile_definitions(multimat_hipcl_FL PRIVATE VERIFY FUSED LINKED)
