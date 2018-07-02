
set(TEST_FILES preprocess_test.cpp)
add_executable(unittest unittest.cpp testing.h ${TEST_FILES})
target_link_libraries(unittest superneurons)
