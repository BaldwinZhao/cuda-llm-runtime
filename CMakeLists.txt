cmake_minimum_required(VERSION 3.24)

project(cuda_llm_runtime LANGUAGES CXX CUDA)

add_executable(cuda_smoke src/smoke_test.cu)

set_target_properties(cuda_smoke PROPERTIES
	CXX_STANDARD 17
	CXX_STANDARD_REQUIRED ON
	CUDA_STANDARD 17
	CUDA_STANDARD_REQUIRED ON
)
