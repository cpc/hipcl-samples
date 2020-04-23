include(CMakeParseArguments)

# Usage:
# add_hipcl_test_or_binary(arguments...)
#
# Arguments:
# NOTEST - bool. if true, only requires EXEC_NAME & SOURCES & ignores all test related arguments
# EXEC_NAME - name of the executable
# SOURCES - sources
# TEST_NAME - name of test (appears in CTest output). if undefined, equals EXEC_NAME
# TEST_PASS - PASS expression. Can be left undefined, then test passes with exit status 0
# TEST_ARGS - arguments to run test executable with

function(add_hipcl_test_or_binary)

  set(NOTEST 0)
  set(options NOTEST)
  set(oneValueArgs EXEC_NAME TEST_NAME TEST_PASS)
  set(multiValueArgs SOURCES TEST_ARGS)

  CMAKE_PARSE_ARGUMENTS(H "${options}" "${oneValueArgs}"
                          "${multiValueArgs}" ${ARGN})

  if(NOT DEFINED H_SOURCES)
    message(FATAL_ERROR "some sources must be given")
  endif()

  if(NOT DEFINED H_EXEC_NAME)
    message(FATAL_ERROR "executable name required")
  endif()

  if((NOT NOTEST) AND (NOT DEFINED H_TEST_NAME))
   set(H_TEST_NAME "${H_EXEC_NAME}")
  endif()

  set_source_files_properties(${H_SOURCES} PROPERTIES LANGUAGE CXX)

  add_executable("${H_EXEC_NAME}" ${H_SOURCES})

  set_target_properties("${H_EXEC_NAME}" PROPERTIES CXX_STANDARD_REQUIRED ON)

  target_link_libraries("${H_EXEC_NAME}" hip::hipcl)

  install(TARGETS "${H_EXEC_NAME}"
          RUNTIME DESTINATION "${INSTALL_BINDIR}")

  if(NOT NOTEST)
    add_test(NAME "${H_TEST_NAME}"
             COMMAND "${CMAKE_CURRENT_BINARY_DIR}/${H_EXEC_NAME}" ${H_TEST_ARGS}
             )

    if(DEFINED H_TEST_PASS)
      set_tests_properties("${H_TEST_NAME}" PROPERTIES
             PASS_REGULAR_EXPRESSION "${H_TEST_PASS}")
    endif()
  endif()
endfunction()

# ARGN = test args
function(add_hipcl_test EXEC_NAME TEST_NAME TEST_PASS SOURCE)

    add_hipcl_test_or_binary(EXEC_NAME "${EXEC_NAME}" TEST_NAME "${TEST_NAME}" TEST_PASS "${TEST_PASS}" SOURCES "${SOURCE}" TEST_ARGS "${ARGN}")

endfunction()


# ARGN = sources
function(add_hipcl_binary EXEC_NAME)

    add_hipcl_test_or_binary(NOTEST EXEC_NAME "${EXEC_NAME}" SOURCES "${ARGN}")

endfunction()


set(SAMPLES
    multimaterial
    BabelStream

    aes
    atomics
    chemv
    clenergy
    memcpy-perf

    mixbench
    PolyBench
    gpu-burn

    montecarlo
    quicksilver

    heat
    mnist

    finance-apps

    nbody
    motionsim
    rng-wallace
    raytrace
    stencil
    vote
)

foreach (SAMPLE ${SAMPLES})
  add_subdirectory(${SAMPLE})
endforeach()

