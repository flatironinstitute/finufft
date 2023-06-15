# platform-indep script to run dumbinputs and check (diff) result
# used by test/CMakeLists.txt

# pick place for stdout to be saved
set(tempout
  "${CMAKE_CURRENT_BINARY_DIR}/dumbinputs${CMAKE_ARGV3}.out"
  )

# pipe dumbinputs there. Is it platform-indep?
# https://stackoverflow.com/questions/36304289/how-to-use-redirection-in-cmake-add-test
execute_process(COMMAND ${CMAKE_CURRENT_BINARY_DIR}/dumbinputs OUTPUT_FILE ${tempout})

# diff the output against reference
execute_process(COMMAND ${CMAKE_COMMAND} -E compare_files ${tempout} ${CMAKE_CURRENT_SOURCE_DIR}/results/dumbinputs${CMAKE_ARGV3}.refout)
