##
#  CMake script for the step-40 tutorial program:
##

# Set the name of the project and target:
set(TARGET "stokes")

# Declare all source files the target consists of. Here, this is only
# the one step-X.cc file, but as you expand your project you may wish
# to add other source files as well. If your project becomes much larger,
# you may want to either replace the following statement by something like
#  file(GLOB_RECURSE TARGET_SRC  "source/*.cc")
#  file(GLOB_RECURSE TARGET_INC  "include/*.h")
#  set(TARGET_SRC ${TARGET_SRC}  ${TARGET_INC})
# or switch altogether to the large project CMakeLists.txt file discussed
# in the "CMake in user projects" page accessible from the "User info"
# page of the documentation.
set(TARGET_SRC
  ${TARGET}.cc
  )

# Usually, you will not need to modify anything beyond this point...

cmake_minimum_required(VERSION 3.13.4)

find_package(deal.II 9.8.0
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
  )
if(NOT ${deal.II_FOUND})
  message(FATAL_ERROR "\n"
    "*** Could not locate a (sufficiently recent) version of deal.II. ***\n\n"
    "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
    "or set an environment variable \"DEAL_II_DIR\" that contains this path."
    )
endif()

#
# Are all dependencies fulfilled?
#
if(DEAL_II_WITH_P4EST) # keep in one line
  message(FATAL_ERROR "
Error! This tutorial requires a deal.II library that was configured with the following options:
    DEAL_II_WITH_P4EST = ON)
endif()

deal_ii_initialize_cached_variables()
set(CLEAN_UP_FILES *.log *.gmv *.gnuplot *.gpl *.eps *.pov *.vtk *.ucd *.d2 *.vtu *.pvtu)
project(${TARGET})
deal_ii_invoke_autopilot()

# --- INDENT ---

add_custom_target(
  indent
  COMMAND clang-format -i stokes.cc
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  COMMENT "Running clang-format ..."
)

# --- TESTING ---

enable_testing()

file(GLOB _tests tests/*.test)

foreach(_test ${_tests})
  set(_test_full ${_test})
  get_filename_component(_testname ${_test} NAME_WE)

  # Read contents of the test file into a variable
  file(READ ${_test_full} _test_file_content)
  
  # Parse mpirun:
  set(mpirun 1) # default value
  string(REGEX MATCH "mpirun=([0-9]+)" _match_mpirun "${_test_file_content}")
  if(_match_mpirun)
    string(REGEX REPLACE "mpirun=([0-9]+)" "\\1" mpirun "${_match_mpirun}")
  endif()
  
  # Parse args
  string(REGEX MATCH "args=([^\n\r]+)" _match_args "${_test_file_content}")
  if(_match_args)
    string(REGEX REPLACE "args=([^\n\r]+)" "\\1" args "${_match_args}")
    string(STRIP "${args}" args)
  else()
    set(args "")
  endif()

  # replace x.test by x.log:
  get_filename_component(_ref_file ${_test} NAME_WE)
  set(_ref_file "${CMAKE_SOURCE_DIR}/tests/${_ref_file}.log")

  message(STATUS "Processing test ${_testname}:")
  message(STATUS "  mpirun=${mpirun}")
  message(STATUS "  args=${args}")
  message(STATUS "  ref=${_ref_file}")


  add_test(NAME ${_testname}
           COMMAND ${CMAKE_COMMAND}
           -D BINARY_DIR=${CMAKE_BINARY_DIR}
           -D SOURCE_DIR=${CMAKE_SOURCE_DIR}
           -D TESTNAME=tests.${_testname}
           -D REF_FILE=${_ref_file}
           -D ARGS=${args}
           -D MPICOUNT=${mpirun}
           -P ${CMAKE_SOURCE_DIR}/tests/run_test.cmake
           WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

  set_tests_properties(${_testname} PROPERTIES TIMEOUT 600)
endforeach()
