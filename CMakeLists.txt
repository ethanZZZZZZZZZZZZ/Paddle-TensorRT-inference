cmake_minimum_required(VERSION 3.18)

project(edge_infer
    VERSION 0.1.0
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(BUILD_TESTS "Build lightweight unit tests" ON)
option(ENABLE_OPENCV "Enable OpenCV video file source" ON)
option(ENABLE_NVTX "Enable NVTX ranges for Nsight Systems profiling" OFF)
option(ENABLE_PADDLE "Enable Paddle Inference backend" OFF)
option(ENABLE_CUDA "Enable CUDA GPU preprocessing module" OFF)
option(ENABLE_TENSORRT_PLUGIN "Build TensorRT YOLO decode + NMS plugin shared library" OFF)
option(ENABLE_PADDLE_GPU_INPUT_SHARE "Enable Paddle Tensor::ShareExternalData GPU input binding path" ON)
set(PADDLE_INFERENCE_DIR "" CACHE PATH "Root directory of a Paddle Inference C++ installation")
set(PADDLE_INFERENCE_INCLUDE_DIR "" CACHE PATH "Directory containing paddle_inference_api.h")
set(PADDLE_INFERENCE_LIB_DIR "" CACHE PATH "Directory containing libpaddle_inference")
set(PADDLE_INFERENCE_LIBRARY "" CACHE FILEPATH "Paddle Inference library file")
set(PADDLE_INFERENCE_THIRD_PARTY_DIR "" CACHE PATH "Paddle Inference third_party or third_party/install directory")
set(PADDLE_INFERENCE_EXTRA_LIB_DIRS "" CACHE STRING "Extra Paddle dependency library directories, separated by semicolons")
set(PADDLE_INFERENCE_EXTRA_LIBS "" CACHE STRING "Extra Paddle dependency libraries, separated by semicolons")
set(TENSORRT_ROOT "" CACHE PATH "Root directory of a TensorRT installation")
set(TENSORRT_INCLUDE_DIR "" CACHE PATH "Directory containing NvInfer.h")
set(TENSORRT_LIBRARY "" CACHE FILEPATH "TensorRT nvinfer library")
set(TENSORRT_PLUGIN_LIBRARY "" CACHE FILEPATH "TensorRT nvinfer_plugin library")

find_package(Threads REQUIRED)

# The project still uses a built-in lightweight YAML subset parser, so yaml-cpp
# is not required. OpenCV is isolated behind ENABLE_OPENCV.
if(ENABLE_OPENCV)
    find_package(OpenCV QUIET)
    if(NOT OpenCV_FOUND)
        message(FATAL_ERROR
            "OpenCV was not found. Install OpenCV or configure with "
            "-DENABLE_OPENCV=OFF to keep the synthetic mock source only."
        )
    endif()
endif()

if(ENABLE_CUDA)
    if(NOT ENABLE_OPENCV)
        message(FATAL_ERROR
            "ENABLE_CUDA=ON currently requires ENABLE_OPENCV=ON because "
            "GpuPreprocessor consumes cv::Mat BGR input."
        )
    endif()
endif()

if(ENABLE_CUDA OR ENABLE_TENSORRT_PLUGIN)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
endif()

if(ENABLE_TENSORRT_PLUGIN)
    set(_TENSORRT_HINT_DIRS
        ${TENSORRT_ROOT}
        "$ENV{TENSORRT_ROOT}"
        "$ENV{TENSORRT_HOME}"
        "$ENV{TensorRT_ROOT}"
        "$ENV{CUDA_HOME}"
        "$ENV{CUDA_PATH}"
        /usr
        /usr/local
        /usr/local/TensorRT
        /usr/local/tensorrt
    )

    if(NOT TENSORRT_INCLUDE_DIR)
        find_path(TENSORRT_INCLUDE_DIR
            NAMES NvInfer.h
            HINTS ${_TENSORRT_HINT_DIRS}
            PATH_SUFFIXES include include/x86_64-linux-gnu include/aarch64-linux-gnu
        )
    endif()

    if(NOT TENSORRT_LIBRARY)
        find_library(TENSORRT_LIBRARY
            NAMES nvinfer
            HINTS ${_TENSORRT_HINT_DIRS}
            PATH_SUFFIXES lib lib64 lib/x64 lib/x86_64-linux-gnu lib/aarch64-linux-gnu targets/x86_64-linux/lib
        )
    endif()

    if(NOT TENSORRT_PLUGIN_LIBRARY)
        find_library(TENSORRT_PLUGIN_LIBRARY
            NAMES nvinfer_plugin
            HINTS ${_TENSORRT_HINT_DIRS}
            PATH_SUFFIXES lib lib64 lib/x64 lib/x86_64-linux-gnu lib/aarch64-linux-gnu targets/x86_64-linux/lib
        )
    endif()

    if(NOT TENSORRT_INCLUDE_DIR OR
       NOT EXISTS "${TENSORRT_INCLUDE_DIR}/NvInfer.h" OR
       NOT TENSORRT_LIBRARY OR
       NOT EXISTS "${TENSORRT_LIBRARY}" OR
       NOT TENSORRT_PLUGIN_LIBRARY OR
       NOT EXISTS "${TENSORRT_PLUGIN_LIBRARY}")
        message(FATAL_ERROR
            "ENABLE_TENSORRT_PLUGIN=ON requires TensorRT headers and libraries. "
            "Set -DTENSORRT_ROOT=/path/to/TensorRT or explicitly set "
            "-DTENSORRT_INCLUDE_DIR=/path/to/include, "
            "-DTENSORRT_LIBRARY=/path/to/libnvinfer.so, and "
            "-DTENSORRT_PLUGIN_LIBRARY=/path/to/libnvinfer_plugin.so. "
            "Configure with -DENABLE_TENSORRT_PLUGIN=OFF to build the main pipeline only."
        )
    endif()

    message(STATUS "TensorRT include dir: ${TENSORRT_INCLUDE_DIR}")
    message(STATUS "TensorRT library: ${TENSORRT_LIBRARY}")
    message(STATUS "TensorRT plugin library: ${TENSORRT_PLUGIN_LIBRARY}")
endif()

if(ENABLE_NVTX)
    find_path(NVTX_INCLUDE_DIR
        NAMES nvToolsExt.h
        PATHS
            "$ENV{CUDA_PATH}"
            "$ENV{CUDA_HOME}"
            "$ENV{CUDA_ROOT}"
        PATH_SUFFIXES include
    )

    find_library(NVTX_LIBRARY
        NAMES nvToolsExt nvToolsExt64_1
        PATHS
            "$ENV{CUDA_PATH}"
            "$ENV{CUDA_HOME}"
            "$ENV{CUDA_ROOT}"
        PATH_SUFFIXES lib lib64 lib/x64
    )

    if(NOT NVTX_INCLUDE_DIR OR NOT NVTX_LIBRARY)
        message(FATAL_ERROR
            "ENABLE_NVTX=ON requires NVIDIA NVTX headers and library. "
            "Install CUDA/Nsight Systems development files, set CUDA_PATH/CUDA_HOME, "
            "or configure with -DENABLE_NVTX=OFF."
        )
    endif()
endif()

if(ENABLE_PADDLE)
    find_package(OpenMP QUIET)

    set(_PADDLE_HINT_DIRS
        ${PADDLE_INFERENCE_DIR}
        "$ENV{PADDLE_INFERENCE_DIR}"
        "$ENV{PADDLE_INFERENCE_HOME}"
    )
    set(_PADDLE_INCLUDE_SUFFIXES
        paddle/include
        paddle/include/paddle
        include
        include/paddle
    )
    set(_PADDLE_LIBRARY_SUFFIXES
        paddle/lib
        paddle/lib64
        lib
        lib64
    )
    set(_PADDLE_LIB_DIR_CANDIDATES "")
    set(_PADDLE_THIRD_PARTY_ROOT_CANDIDATES "")
    set(_PADDLE_SEARCH_ROOTS "")

    foreach(_paddle_root IN LISTS _PADDLE_HINT_DIRS)
        if(_paddle_root)
            if(IS_DIRECTORY "${_paddle_root}")
                list(APPEND _PADDLE_SEARCH_ROOTS "${_paddle_root}")
            endif()
            list(APPEND _PADDLE_LIB_DIR_CANDIDATES
                "${_paddle_root}/paddle/lib"
                "${_paddle_root}/paddle/lib64"
                "${_paddle_root}/lib"
                "${_paddle_root}/lib64"
            )
            list(APPEND _PADDLE_THIRD_PARTY_ROOT_CANDIDATES
                "${_paddle_root}/third_party"
                "${_paddle_root}/third_party/install"
            )
            list(APPEND _PADDLE_LIB_DIR_CANDIDATES
                "${_paddle_root}/third_party/install/onednn/lib"
                "${_paddle_root}/third_party/install/mkldnn/lib"
                "${_paddle_root}/third_party/install/mklml/lib"
                "${_paddle_root}/third_party/install/glog/lib"
                "${_paddle_root}/third_party/install/gflags/lib"
                "${_paddle_root}/third_party/install/protobuf/lib"
                "${_paddle_root}/third_party/install/xxhash/lib"
                "${_paddle_root}/third_party/install/utf8proc/lib"
                "${_paddle_root}/third_party/install/cryptopp/lib"
            )
        endif()
    endforeach()

    if(PADDLE_INFERENCE_LIB_DIR)
        list(PREPEND _PADDLE_LIB_DIR_CANDIDATES "${PADDLE_INFERENCE_LIB_DIR}")
        if(IS_DIRECTORY "${PADDLE_INFERENCE_LIB_DIR}")
            list(APPEND _PADDLE_SEARCH_ROOTS "${PADDLE_INFERENCE_LIB_DIR}")
        endif()
    endif()

    if(PADDLE_INFERENCE_THIRD_PARTY_DIR)
        list(PREPEND _PADDLE_THIRD_PARTY_ROOT_CANDIDATES "${PADDLE_INFERENCE_THIRD_PARTY_DIR}")
        list(APPEND _PADDLE_LIB_DIR_CANDIDATES
            "${PADDLE_INFERENCE_THIRD_PARTY_DIR}/install/onednn/lib"
            "${PADDLE_INFERENCE_THIRD_PARTY_DIR}/install/mkldnn/lib"
            "${PADDLE_INFERENCE_THIRD_PARTY_DIR}/install/mklml/lib"
            "${PADDLE_INFERENCE_THIRD_PARTY_DIR}/install/glog/lib"
            "${PADDLE_INFERENCE_THIRD_PARTY_DIR}/install/gflags/lib"
            "${PADDLE_INFERENCE_THIRD_PARTY_DIR}/install/protobuf/lib"
            "${PADDLE_INFERENCE_THIRD_PARTY_DIR}/install/xxhash/lib"
            "${PADDLE_INFERENCE_THIRD_PARTY_DIR}/install/utf8proc/lib"
            "${PADDLE_INFERENCE_THIRD_PARTY_DIR}/install/cryptopp/lib"
            "${PADDLE_INFERENCE_THIRD_PARTY_DIR}/onednn/lib"
            "${PADDLE_INFERENCE_THIRD_PARTY_DIR}/mkldnn/lib"
            "${PADDLE_INFERENCE_THIRD_PARTY_DIR}/mklml/lib"
            "${PADDLE_INFERENCE_THIRD_PARTY_DIR}/glog/lib"
            "${PADDLE_INFERENCE_THIRD_PARTY_DIR}/gflags/lib"
            "${PADDLE_INFERENCE_THIRD_PARTY_DIR}/protobuf/lib"
            "${PADDLE_INFERENCE_THIRD_PARTY_DIR}/xxhash/lib"
            "${PADDLE_INFERENCE_THIRD_PARTY_DIR}/utf8proc/lib"
            "${PADDLE_INFERENCE_THIRD_PARTY_DIR}/cryptopp/lib"
        )
    endif()

    if(PADDLE_INFERENCE_INCLUDE_DIR AND IS_DIRECTORY "${PADDLE_INFERENCE_INCLUDE_DIR}")
        list(APPEND _PADDLE_SEARCH_ROOTS "${PADDLE_INFERENCE_INCLUDE_DIR}")
    endif()

    if(_PADDLE_SEARCH_ROOTS)
        list(REMOVE_DUPLICATES _PADDLE_SEARCH_ROOTS)
    endif()

    if(NOT PADDLE_INFERENCE_INCLUDE_DIR)
        find_path(PADDLE_INFERENCE_INCLUDE_DIR
            NAMES paddle_inference_api.h
            HINTS ${_PADDLE_HINT_DIRS}
            PATH_SUFFIXES ${_PADDLE_INCLUDE_SUFFIXES}
        )
    endif()

    if(NOT PADDLE_INFERENCE_INCLUDE_DIR OR
       NOT EXISTS "${PADDLE_INFERENCE_INCLUDE_DIR}/paddle_inference_api.h")
        foreach(_paddle_search_root IN LISTS _PADDLE_SEARCH_ROOTS)
            file(GLOB_RECURSE _PADDLE_HEADER_MATCHES
                "${_paddle_search_root}/paddle_inference_api.h"
            )
            if(_PADDLE_HEADER_MATCHES)
                list(SORT _PADDLE_HEADER_MATCHES)
                list(GET _PADDLE_HEADER_MATCHES 0 _PADDLE_HEADER_FILE)
                get_filename_component(_PADDLE_HEADER_DIR "${_PADDLE_HEADER_FILE}" DIRECTORY)
                set(PADDLE_INFERENCE_INCLUDE_DIR "${_PADDLE_HEADER_DIR}"
                    CACHE PATH "Directory containing paddle_inference_api.h" FORCE)
                break()
            endif()
        endforeach()
    endif()

    if(PADDLE_INFERENCE_INCLUDE_DIR AND
       NOT EXISTS "${PADDLE_INFERENCE_INCLUDE_DIR}/paddle_inference_api.h")
        message(FATAL_ERROR
            "PADDLE_INFERENCE_INCLUDE_DIR does not contain paddle_inference_api.h: "
            "${PADDLE_INFERENCE_INCLUDE_DIR}"
        )
    endif()

    if(NOT PADDLE_INFERENCE_LIBRARY)
        find_library(PADDLE_INFERENCE_LIBRARY
            NAMES paddle_inference libpaddle_inference
            HINTS
                ${PADDLE_INFERENCE_LIB_DIR}
                ${_PADDLE_HINT_DIRS}
            PATH_SUFFIXES ${_PADDLE_LIBRARY_SUFFIXES}
        )
    endif()

    if(NOT PADDLE_INFERENCE_LIBRARY OR NOT EXISTS "${PADDLE_INFERENCE_LIBRARY}")
        foreach(_paddle_search_root IN LISTS _PADDLE_SEARCH_ROOTS)
            file(GLOB_RECURSE _PADDLE_LIBRARY_MATCHES
                "${_paddle_search_root}/libpaddle_inference.so"
            )
            if(NOT _PADDLE_LIBRARY_MATCHES)
                file(GLOB_RECURSE _PADDLE_LIBRARY_MATCHES
                    "${_paddle_search_root}/libpaddle_inference.so*"
                )
            endif()
            if(_PADDLE_LIBRARY_MATCHES)
                list(SORT _PADDLE_LIBRARY_MATCHES)
                list(GET _PADDLE_LIBRARY_MATCHES 0 _PADDLE_LIBRARY_FILE)
                set(PADDLE_INFERENCE_LIBRARY "${_PADDLE_LIBRARY_FILE}"
                    CACHE FILEPATH "Paddle Inference library file" FORCE)
                get_filename_component(PADDLE_INFERENCE_LIB_DIR "${PADDLE_INFERENCE_LIBRARY}" DIRECTORY)
                set(PADDLE_INFERENCE_LIB_DIR "${PADDLE_INFERENCE_LIB_DIR}"
                    CACHE PATH "Directory containing libpaddle_inference" FORCE)
                break()
            endif()
        endforeach()
    endif()

    if(PADDLE_INFERENCE_LIBRARY AND NOT EXISTS "${PADDLE_INFERENCE_LIBRARY}")
        message(FATAL_ERROR
            "PADDLE_INFERENCE_LIBRARY does not exist: ${PADDLE_INFERENCE_LIBRARY}"
        )
    endif()

    if(PADDLE_INFERENCE_LIBRARY)
        get_filename_component(_PADDLE_MAIN_LIB_DIR "${PADDLE_INFERENCE_LIBRARY}" DIRECTORY)
        list(PREPEND _PADDLE_LIB_DIR_CANDIDATES "${_PADDLE_MAIN_LIB_DIR}")
    endif()

    if(NOT PADDLE_INFERENCE_INCLUDE_DIR OR NOT PADDLE_INFERENCE_LIBRARY)
        message(FATAL_ERROR
            "ENABLE_PADDLE=ON requires Paddle Inference C++ headers and library. "
            "Current PADDLE_INFERENCE_DIR='${PADDLE_INFERENCE_DIR}', "
            "PADDLE_INFERENCE_INCLUDE_DIR='${PADDLE_INFERENCE_INCLUDE_DIR}', "
            "PADDLE_INFERENCE_LIB_DIR='${PADDLE_INFERENCE_LIB_DIR}', "
            "PADDLE_INFERENCE_LIBRARY='${PADDLE_INFERENCE_LIBRARY}'. "
            "Set -DPADDLE_INFERENCE_DIR=/workspace/3rd/paddle_inference or explicitly set "
            "-DPADDLE_INFERENCE_INCLUDE_DIR to the directory that contains paddle_inference_api.h "
            "and -DPADDLE_INFERENCE_LIBRARY to the full libpaddle_inference.so path. "
            "You can inspect the package with: "
            "find ${PADDLE_INFERENCE_DIR} -name paddle_inference_api.h -o -name libpaddle_inference.so. "
            "Configure with -DENABLE_PADDLE=OFF to keep the mock backend only."
        )
    endif()

    set(_PADDLE_DEP_LIB_DIRS "")
    set(_PADDLE_DEP_LIBS "")

    foreach(_lib_dir IN LISTS _PADDLE_LIB_DIR_CANDIDATES PADDLE_INFERENCE_EXTRA_LIB_DIRS)
        if(_lib_dir AND IS_DIRECTORY "${_lib_dir}")
            list(APPEND _PADDLE_DEP_LIB_DIRS "${_lib_dir}")
            file(GLOB _PADDLE_RUNTIME_LIBS
                "${_lib_dir}/libphi*.so"
                "${_lib_dir}/libpaddle_*.so"
                "${_lib_dir}/libcommon.so"
            )
            list(APPEND _PADDLE_DEP_LIBS ${_PADDLE_RUNTIME_LIBS})
        endif()
    endforeach()

    foreach(_third_party_root IN LISTS _PADDLE_THIRD_PARTY_ROOT_CANDIDATES)
        if(_third_party_root AND IS_DIRECTORY "${_third_party_root}")
            get_filename_component(_third_party_dir_name "${_third_party_root}" NAME)
            if(_third_party_dir_name STREQUAL "lib" OR _third_party_dir_name STREQUAL "lib64")
                list(APPEND _PADDLE_DEP_LIB_DIRS "${_third_party_root}")
            endif()

            file(GLOB_RECURSE _PADDLE_THIRD_PARTY_LIB_DIRS
                LIST_DIRECTORIES true
                "${_third_party_root}/*/lib"
                "${_third_party_root}/*/lib64"
            )
            foreach(_third_party_lib_dir IN LISTS _PADDLE_THIRD_PARTY_LIB_DIRS)
                if(IS_DIRECTORY "${_third_party_lib_dir}")
                    list(APPEND _PADDLE_DEP_LIB_DIRS "${_third_party_lib_dir}")
                endif()
            endforeach()

            file(GLOB_RECURSE _PADDLE_RECURSIVE_DEP_LIBS
                "${_third_party_root}/libdnnl.so*"
                "${_third_party_root}/libmkldnn.so*"
                "${_third_party_root}/libiomp5.so*"
                "${_third_party_root}/libmklml_intel.so*"
                "${_third_party_root}/libmklml_gnu.so*"
                "${_third_party_root}/libgomp.so*"
                "${_third_party_root}/libglog.so*"
                "${_third_party_root}/libgflags.so*"
                "${_third_party_root}/libprotobuf.so*"
                "${_third_party_root}/libxxhash.so*"
                "${_third_party_root}/libutf8proc.so*"
                "${_third_party_root}/libcryptopp.so*"
            )
            foreach(_recursive_dep_lib IN LISTS _PADDLE_RECURSIVE_DEP_LIBS)
                get_filename_component(_recursive_dep_dir "${_recursive_dep_lib}" DIRECTORY)
                list(APPEND _PADDLE_DEP_LIB_DIRS "${_recursive_dep_dir}")
            endforeach()
            list(APPEND _PADDLE_DEP_LIBS ${_PADDLE_RECURSIVE_DEP_LIBS})
        endif()
    endforeach()

    foreach(_dep_lib_dir IN LISTS _PADDLE_DEP_LIB_DIRS)
        file(GLOB _PADDLE_KNOWN_DEP_LIBS
            "${_dep_lib_dir}/libdnnl.so*"
            "${_dep_lib_dir}/libmkldnn.so*"
            "${_dep_lib_dir}/libiomp5.so*"
            "${_dep_lib_dir}/libmklml_intel.so*"
            "${_dep_lib_dir}/libmklml_gnu.so*"
            "${_dep_lib_dir}/libgomp.so*"
            "${_dep_lib_dir}/libglog.so*"
            "${_dep_lib_dir}/libgflags.so*"
            "${_dep_lib_dir}/libprotobuf.so*"
            "${_dep_lib_dir}/libxxhash.so*"
            "${_dep_lib_dir}/libutf8proc.so*"
            "${_dep_lib_dir}/libcryptopp.so*"
        )
        list(APPEND _PADDLE_DEP_LIBS ${_PADDLE_KNOWN_DEP_LIBS})
    endforeach()

    list(APPEND _PADDLE_DEP_LIBS ${PADDLE_INFERENCE_EXTRA_LIBS})
    list(REMOVE_ITEM _PADDLE_DEP_LIBS "${PADDLE_INFERENCE_LIBRARY}")

    if(_PADDLE_DEP_LIB_DIRS)
        list(REMOVE_DUPLICATES _PADDLE_DEP_LIB_DIRS)
    endif()
    if(_PADDLE_DEP_LIBS)
        list(REMOVE_DUPLICATES _PADDLE_DEP_LIBS)
    endif()

    set(PADDLE_INFERENCE_RESOLVED_LIB_DIRS ${_PADDLE_DEP_LIB_DIRS})
    set(PADDLE_INFERENCE_RESOLVED_LIBS ${_PADDLE_DEP_LIBS})

    message(STATUS "Paddle Inference include dir: ${PADDLE_INFERENCE_INCLUDE_DIR}")
    message(STATUS "Paddle Inference library: ${PADDLE_INFERENCE_LIBRARY}")
    message(STATUS "Paddle Inference dependency lib dirs: ${PADDLE_INFERENCE_RESOLVED_LIB_DIRS}")
    message(STATUS "Paddle Inference dependency libs: ${PADDLE_INFERENCE_RESOLVED_LIBS}")
endif()

add_library(edge_infer_core
    src/common/config.cpp
    src/pipeline/batch_scheduler.cpp
    src/pipeline/video_source.cpp
    src/preprocess/preprocessor.cpp
    src/inference/mock_infer_engine.cpp
    src/inference/predictor_pool.cpp
    src/postprocess/postprocessor.cpp
    src/profiling/profiler.cpp
    src/pipeline/pipeline.cpp
)

if(ENABLE_PADDLE)
    target_sources(edge_infer_core
        PRIVATE
            src/inference/paddle_infer_engine.cpp
    )
endif()

if(ENABLE_CUDA OR ENABLE_TENSORRT_PLUGIN)
    target_sources(edge_infer_core
        PRIVATE
            src/cuda/cuda_runtime_utils.cpp
    )
endif()

if(ENABLE_CUDA)
    target_sources(edge_infer_core
        PRIVATE
            src/preprocess/gpu_preprocessor.cpp
            src/postprocess/gpu_decode_pre_nms.cpp
            cuda/preprocess_kernel.cu
            cuda/decode_kernel.cu
            cuda/nms_kernel.cu
    )
    set_target_properties(edge_infer_core
        PROPERTIES
            CUDA_STANDARD 17
            CUDA_STANDARD_REQUIRED ON
    )
endif()

if(ENABLE_TENSORRT_PLUGIN)
    target_sources(edge_infer_core
        PRIVATE
            src/postprocess/trt_postprocess_engine.cpp
            plugins/yolo_decode_plugin/yolo_decode_plugin.cpp
    )
    if(NOT ENABLE_CUDA)
        target_sources(edge_infer_core
            PRIVATE
                cuda/decode_kernel.cu
                cuda/nms_kernel.cu
        )
    endif()
    set_target_properties(edge_infer_core
        PROPERTIES
            CUDA_STANDARD 17
            CUDA_STANDARD_REQUIRED ON
    )
endif()

target_include_directories(edge_infer_core
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

target_compile_options(edge_infer_core
    PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>
        $<$<CXX_COMPILER_ID:MSVC>:/W4>
)

target_link_libraries(edge_infer_core
    PUBLIC
        Threads::Threads
)

if(ENABLE_OPENCV)
    target_compile_definitions(edge_infer_core
        PUBLIC
            EDGE_ENABLE_OPENCV=1
    )
    target_include_directories(edge_infer_core
        SYSTEM PUBLIC
            ${OpenCV_INCLUDE_DIRS}
    )
    target_link_libraries(edge_infer_core
        PUBLIC
            ${OpenCV_LIBS}
    )
endif()

if(ENABLE_NVTX)
    target_compile_definitions(edge_infer_core
        PUBLIC
            EDGE_ENABLE_NVTX=1
    )
    target_include_directories(edge_infer_core
        SYSTEM PUBLIC
            ${NVTX_INCLUDE_DIR}
    )
    target_link_libraries(edge_infer_core
        PUBLIC
            ${NVTX_LIBRARY}
    )
endif()

if(ENABLE_PADDLE)
    target_compile_definitions(edge_infer_core
        PUBLIC
            EDGE_ENABLE_PADDLE=1
    )
    if(ENABLE_PADDLE_GPU_INPUT_SHARE)
        target_compile_definitions(edge_infer_core
            PUBLIC
                EDGE_ENABLE_PADDLE_GPU_INPUT_SHARE=1
        )
    endif()
    target_include_directories(edge_infer_core
        SYSTEM PUBLIC
            ${PADDLE_INFERENCE_INCLUDE_DIR}
    )
    target_link_libraries(edge_infer_core
        PUBLIC
            ${PADDLE_INFERENCE_LIBRARY}
            ${PADDLE_INFERENCE_RESOLVED_LIBS}
            ${CMAKE_DL_LIBS}
    )
    if(OpenMP_CXX_FOUND)
        target_link_libraries(edge_infer_core
            PUBLIC
                OpenMP::OpenMP_CXX
        )
    endif()
    if(PADDLE_INFERENCE_RESOLVED_LIB_DIRS)
        target_link_directories(edge_infer_core
            PUBLIC
                ${PADDLE_INFERENCE_RESOLVED_LIB_DIRS}
        )
        foreach(_paddle_rpath_dir IN LISTS PADDLE_INFERENCE_RESOLVED_LIB_DIRS)
            target_link_options(edge_infer_core
                PUBLIC
                    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wl,-rpath-link,${_paddle_rpath_dir}>
                    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wl,-rpath,${_paddle_rpath_dir}>
            )
        endforeach()
    endif()
endif()

if(ENABLE_CUDA)
    target_compile_definitions(edge_infer_core
        PUBLIC
            EDGE_ENABLE_CUDA=1
    )
    target_link_libraries(edge_infer_core
        PUBLIC
            CUDA::cudart
    )
endif()

if(ENABLE_TENSORRT_PLUGIN)
    target_compile_definitions(edge_infer_core
        PUBLIC
            EDGE_ENABLE_TENSORRT_PLUGIN=1
    )
    target_include_directories(edge_infer_core
        PUBLIC
            ${PROJECT_SOURCE_DIR}/plugins/yolo_decode_plugin
    )
    target_include_directories(edge_infer_core
        SYSTEM PUBLIC
            ${TENSORRT_INCLUDE_DIR}
    )
    target_link_libraries(edge_infer_core
        PUBLIC
            CUDA::cudart
            ${TENSORRT_LIBRARY}
            ${TENSORRT_PLUGIN_LIBRARY}
    )
endif()

add_executable(edge_infer
    src/main.cpp
)

target_link_libraries(edge_infer
    PRIVATE
        edge_infer_core
)

if(ENABLE_TENSORRT_PLUGIN)
    add_library(yolo_decode_plugin SHARED
        plugins/yolo_decode_plugin/yolo_decode_plugin.cpp
        cuda/decode_kernel.cu
        cuda/nms_kernel.cu
    )

    target_compile_definitions(yolo_decode_plugin
        PRIVATE
            EDGE_ENABLE_CUDA=1
    )

    target_include_directories(yolo_decode_plugin
        PRIVATE
            ${PROJECT_SOURCE_DIR}/include
            ${PROJECT_SOURCE_DIR}/plugins/yolo_decode_plugin
    )

    target_include_directories(yolo_decode_plugin
        SYSTEM PRIVATE
            ${TENSORRT_INCLUDE_DIR}
    )

    target_link_libraries(yolo_decode_plugin
        PRIVATE
            CUDA::cudart
            ${TENSORRT_LIBRARY}
            ${TENSORRT_PLUGIN_LIBRARY}
    )

    set_target_properties(yolo_decode_plugin
        PROPERTIES
            CXX_STANDARD 17
            CXX_STANDARD_REQUIRED ON
            CUDA_STANDARD 17
            CUDA_STANDARD_REQUIRED ON
    )
endif()

if(ENABLE_PADDLE AND PADDLE_INFERENCE_RESOLVED_LIB_DIRS)
    set_target_properties(edge_infer
        PROPERTIES
            SKIP_BUILD_RPATH FALSE
            BUILD_WITH_INSTALL_RPATH FALSE
            BUILD_RPATH "${PADDLE_INFERENCE_RESOLVED_LIB_DIRS}"
            INSTALL_RPATH "${PADDLE_INFERENCE_RESOLVED_LIB_DIRS}"
    )
endif()

if(BUILD_TESTS)
    enable_testing()

    add_executable(test_mock_engine
        tests/test_mock_engine.cpp
    )

    target_link_libraries(test_mock_engine
        PRIVATE
            edge_infer_core
    )

    add_test(NAME test_mock_engine COMMAND test_mock_engine)

    add_executable(test_postprocess_transform
        tests/test_postprocess_transform.cpp
    )

    target_link_libraries(test_postprocess_transform
        PRIVATE
            edge_infer_core
    )

    add_test(NAME test_postprocess_transform COMMAND test_postprocess_transform)

    add_executable(test_postprocess_nms
        tests/test_postprocess_nms.cpp
    )

    target_link_libraries(test_postprocess_nms
        PRIVATE
            edge_infer_core
    )

    add_test(NAME test_postprocess_nms COMMAND test_postprocess_nms)

    add_executable(test_batch_scheduler
        tests/test_batch_scheduler.cpp
    )

    target_link_libraries(test_batch_scheduler
        PRIVATE
            edge_infer_core
    )

    add_test(NAME test_batch_scheduler COMMAND test_batch_scheduler)

    if(ENABLE_OPENCV)
        add_executable(test_cpu_preprocess
            tests/test_cpu_preprocess.cpp
        )

        target_link_libraries(test_cpu_preprocess
            PRIVATE
                edge_infer_core
        )

        add_test(NAME test_cpu_preprocess COMMAND test_cpu_preprocess)
    endif()

    if(ENABLE_CUDA)
        add_executable(test_gpu_preprocess
            tests/test_gpu_preprocess.cpp
        )

        target_link_libraries(test_gpu_preprocess
            PRIVATE
                edge_infer_core
        )

        add_test(NAME test_gpu_preprocess COMMAND test_gpu_preprocess)

        add_executable(test_gpu_decode_pre_nms
            tests/test_gpu_decode_pre_nms.cpp
        )

        target_link_libraries(test_gpu_decode_pre_nms
            PRIVATE
                edge_infer_core
        )

        add_test(NAME test_gpu_decode_pre_nms COMMAND test_gpu_decode_pre_nms)

        add_executable(test_gpu_nms
            tests/test_gpu_nms.cpp
        )

        target_link_libraries(test_gpu_nms
            PRIVATE
                edge_infer_core
        )

        add_test(NAME test_gpu_nms COMMAND test_gpu_nms)
    endif()
endif()
