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
execute_process(COMMAND ${CMAKE_COMMAND} -E compare_files --ignore-eol ${tempout} ${CMAKE_CURRENT_SOURCE_DIR}/results/dumbinputs${CMAKE_ARGV3}.refout)



# ================== the above was an example extra .cmake script
# in an attempt at platform-indep.

# here are other ideas from the CMakeLists.txt by Alex, and Libin:

if(0)
# Here's where to save output of dumbinputs for later diff
set(tempout
  "${CMAKE_CURRENT_BINARY_DIR}/dumbinputs${SUFFIX}.out"
  )
# follow https://stackoverflow.com/questions/36304289/how-to-use-redirection-in-cmake-add-test
# https://stackoverflow.com/questions/39960173/run-custom-shell-script-with-cmake
# Is pipe here platform-indep? very hard to find out. sh or bash are not cross-platform :(
add_custom_target(
  run_dumbinputs_${PREC} ALL
  COMMENT "Custom target run of dumbinputs${SUFFIX}..."
  # this still spews stuff to stderr which looks broken to the user; it is not...
  COMMAND OMP_NUM_THREADS=1 ${CMAKE_CURRENT_BINARY_DIR}/dumbinputs${SUFFIX} > ${tempout}
  # I don't know if correctly using dependency: it needs the current executable...
  DEPENDS dumbinputs${SUFFIX}
  )
# I considered add_test() as another way (fails since can't handle the pipe >)
# and add_custom_command(), which fails to run unless add_custom_target used anyway.
# annoyingly, add_test COMMAND is quite lame: does not allow piping, as above,
# or multiple cmds
# See: https://stackoverflow.com/questions/3065220/ctest-with-multiple-commands?rq=4
add_test(
  NAME check_dumbinputs_${PREC}
  COMMAND ${CMAKE_COMMAND} -E compare_files ${tempout} ${CMAKE_CURRENT_SOURCE_DIR}/results/dumbinputs${SUFFIX}.refout
  )
endif()

# another way of using add_test, remove add_custom_target by calling the devel/check_dumbinputs.cmake
# if move check_dumbinputs.cmake to test directory, ${CMAKE_SOURCE_DIR} should be changed to ${CMAKE_CURRENT_SOURCE_DIR}
add_test(
  NAME check_dumbinputs_${PREC}
  COMMAND ${CMAKE_COMMAND} -P ${CMAKE_SOURCE_DIR}/devel/check_dumbinputs.cmake ${SUFFIX}
  )
