cmake_minimum_required(VERSION 3.24)

project(cuda_llm_runtime LANGUAGES CXX CUDA)

message(STATUS "CXX Compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "CUDA Compiler: ${CMAKE_CUDA_COMPILER}")
message(STATUS "CUDA Host Compiler: ${CMAKE_CUDA_HOST_COMPILER}")
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")


add_executable(cuda_smoke
	src/smoke_test.cu
	src/check_error.cu
)

set_target_properties(cuda_smoke PROPERTIES
	CXX_STANDARD 17
	CXX_STANDARD_REQUIRED ON
	CUDA_STANDARD 17
	CUDA_STANDARD_REQUIRED ON
)
