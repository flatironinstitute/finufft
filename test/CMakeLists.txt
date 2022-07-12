
file(GLOB test_src "*.cu")

foreach(srcfile ${test_src})
  string(REPLACE ".cu" "" executable ${srcfile})
  get_filename_component(executable ${executable} NAME)
  add_executable(${executable} ${srcfile})
  target_include_directories(${executable} PUBLIC ${CUFINUFFT_INCLUDE_DIRS})
  target_link_libraries(${executable} PUBLIC cufinufft_static)
endforeach()
