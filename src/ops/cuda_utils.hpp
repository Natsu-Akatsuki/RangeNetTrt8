#ifndef CUDA_UTILS_HPP
#define CUDA_UTILS_HPP

#include <cuda_runtime_api.h>
#include <memory>
#include <stdio.h>
#include <stdlib.h>

static void CheckCudaError(cudaError_t err, const char *file, int line) {
  if (err != cudaSuccess) {
    printf("[%s@%d]%s in %s:%d\n", cudaGetErrorName(err), err,
           cudaGetErrorString(err), file, line);
    exit(EXIT_FAILURE);
  }
}
#define CHECK_CUDA_ERROR(err) (CheckCudaError(err, __FILE__, __LINE__))

namespace cuda {
/**
 * @brief 自定义内存回收逻辑
 */
struct deleter_gpu {
  void operator()(void *p) const { CHECK_CUDA_ERROR(::cudaFree(p)); }
};

struct deleter_pin {
  void operator()(void *p) const { CHECK_CUDA_ERROR(::cudaFreeHost(p)); }
};

template<typename T> using unique_gpu_ptr = std::unique_ptr<T, deleter_gpu>;
template<typename T> using unique_pin_ptr = std::unique_ptr<T, deleter_pin>;

// array type for gpu
template<typename T>
typename std::enable_if<std::is_array<T>::value, cuda::unique_gpu_ptr<T>>::type
make_gpu_unique(const std::size_t n, bool isMulTypeSize = true) {
  // e.g typename std::remove_extent<float[]>::type -> float;
  // 取得数组中元素的类型
  using U = typename std::remove_extent<T>::type;
  U *p;
  if (isMulTypeSize) {
    CHECK_CUDA_ERROR(
        ::cudaMalloc(reinterpret_cast<void **>(&p), sizeof(U) * n));
  } else {
    CHECK_CUDA_ERROR(::cudaMallocHost(reinterpret_cast<void **>(&p), n));
  }
  return cuda::unique_gpu_ptr<T>{p};
}

// array type for pinned memory
template<typename T>
typename std::enable_if<std::is_array<T>::value, cuda::unique_pin_ptr<T>>::type
make_pin_unique(const std::size_t n, bool isMulTypeSize = true) {
  // e.g typename std::remove_extent<float[]>::type -> float;
  // 取得数组中元素的类型
  using U = typename std::remove_extent<T>::type;
  U *p;
  if (isMulTypeSize) {
    CHECK_CUDA_ERROR(
        ::cudaMallocHost(reinterpret_cast<void **>(&p), sizeof(U) * n));
  } else {
    CHECK_CUDA_ERROR(::cudaMallocHost(reinterpret_cast<void **>(&p), n));
  }
  return cuda::unique_pin_ptr<T>{p};
}

#if 0
// 普通类型
template <typename T> cuda::unique_ptr<T> make_unique() {
  T *p;
  CHECK_CUDA_ERROR(::cudaMalloc(reinterpret_cast<void **>(&p), sizeof(T)));
  return cuda::unique_ptr<T>{p};
}
#endif /*code block*/

} // namespace cuda
#endif // CUDA_UTILS_HPP