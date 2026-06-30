cmake_minimum_required(VERSION 3.10)
project(dronedet CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# ncnn_DIR 로 빌드별 ncnn(install) 지정:
#   host : -Dncnn_DIR=<ncnn>/build-host/install/lib/cmake/ncnn
#   ml2  : -Dncnn_DIR=<ncnn>/build-ml2/install/lib/cmake/ncnn  (+ NDK toolchain)
find_package(ncnn REQUIRED)

# 추론 모듈. ML2 배포는 -DDRONEDET_SHARED=ON 으로 libdronedet.so 생성.
option(DRONEDET_SHARED "build dronedet as a shared library (.so)" OFF)
if(DRONEDET_SHARED)
  add_library(dronedet SHARED drone_detector.cpp)
else()
  add_library(dronedet STATIC drone_detector.cpp)
endif()
target_link_libraries(dronedet PUBLIC ncnn)
target_include_directories(dronedet PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# self-test 바이너리(호스트=4090 Vulkan 검증, ML2=on-device runbook 에서 실행).
add_executable(dronedet_selftest test_host.cpp)
target_link_libraries(dronedet_selftest PRIVATE dronedet)

# Android(NDK)에서는 Vulkan 로더를 명시 링크.
if(ANDROID)
  find_library(vulkan-lib vulkan)
  if(vulkan-lib)
    target_link_libraries(dronedet PUBLIC ${vulkan-lib})
  endif()
endif()
