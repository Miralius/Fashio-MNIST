cmake_minimum_required(VERSION 3.20)

set(PATCH_VERSION "1" CACHE INTERNAL "Patch version")
set(PROJECT_VERSION_VALUE 0.0.${PATCH_VERSION})

project(fashio_mnist VERSION ${PROJECT_VERSION_VALUE} LANGUAGES CXX)

include(GNUInstallDirs)

option(BUILD_TESTING "Build unit tests" ON)

find_path(TENSORFLOW_C_INCLUDE_DIR
        NAMES tensorflow/c/c_api.h
        HINTS /usr/local/include
        DOC "Directory containing the TensorFlow C API headers"
)
find_library(TENSORFLOW_C_LIBRARY
        NAMES tensorflow
        HINTS /usr/local/lib
        DOC "TensorFlow C API library"
)

if(NOT TENSORFLOW_C_INCLUDE_DIR OR NOT TENSORFLOW_C_LIBRARY)
    message(FATAL_ERROR
            "TensorFlow C API was not found. Set TENSORFLOW_C_INCLUDE_DIR and "
            "TENSORFLOW_C_LIBRARY explicitly.")
endif()

add_library(fashion_mnist_lib
        src/csv_reader.cpp
        src/tensorflow_classifier.cpp
)
target_include_directories(fashion_mnist_lib
        PUBLIC include
        PRIVATE ${TENSORFLOW_C_INCLUDE_DIR}
)
target_link_libraries(fashion_mnist_lib PUBLIC ${TENSORFLOW_C_LIBRARY})
target_compile_features(fashion_mnist_lib PUBLIC cxx_std_23)

add_executable(fashio_mnist main.cpp)
target_link_libraries(fashio_mnist PRIVATE fashion_mnist_lib)

function(enable_project_warnings target_name)
    if (MSVC)
        target_compile_options(${target_name} PRIVATE /W4 /WX)
    else ()
        target_compile_options(${target_name} PRIVATE -Wall -Wextra -pedantic -Werror)
    endif()
endfunction()

target_compile_features(fashio_mnist PRIVATE cxx_std_23)
enable_project_warnings(fashion_mnist_lib)
enable_project_warnings(fashio_mnist)

install(TARGETS fashio_mnist
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

if(BUILD_TESTING)
    enable_testing()
    find_package(GTest REQUIRED)

    add_executable(unit_tests EXCLUDE_FROM_ALL
            tests/main.cpp
    )
    target_link_libraries(unit_tests PRIVATE fashion_mnist_lib GTest::GTest)
    target_compile_features(unit_tests PRIVATE cxx_std_23)
    enable_project_warnings(unit_tests)

    add_test(NAME Tests COMMAND unit_tests)
endif()

set(CPACK_GENERATOR DEB)
set(CPACK_PACKAGE_VERSION_MAJOR "${PROJECT_VERSION_MAJOR}")
set(CPACK_PACKAGE_VERSION_MINOR "${PROJECT_VERSION_MINOR}")
set(CPACK_PACKAGE_VERSION_PATCH "${PROJECT_VERSION_PATCH}")
set(CPACK_PACKAGE_CONTACT example@example.com)
include(CPack)
