if(FINUFFT_USE_CPU)
    install(TARGETS finufft LIBRARY DESTINATION finufft)
endif()

if(FINUFFT_USE_GPU)
    install(TARGETS finufft_gpu LIBRARY DESTINATION cufinufft)
endif()
