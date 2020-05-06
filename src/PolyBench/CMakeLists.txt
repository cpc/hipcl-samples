set(SAMPLES
    covariance
    correlation
    atax    
    convolution-2d  
    gemm     
    gramschmidt      
    lu     
    syrk
    2mm             
    3mm             
    bicg    
    convolution-3d  
    doitgen     
    gemver   
    jacobi-1d-imper  
    mvt
    adi        
    fdtd-2d     
    gesummv  
    jacobi-2d-imper  
    syr2k
)

foreach (SAMPLE ${SAMPLES})
  add_subdirectory(${SAMPLE})
endforeach()

