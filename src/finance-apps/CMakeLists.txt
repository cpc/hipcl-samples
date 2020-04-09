set(SAMPLES BlackScholesEngine  BondsEngine  RepoEngine)

foreach (SAMPLE ${SAMPLES})
  add_subdirectory(${SAMPLE})
endforeach()

