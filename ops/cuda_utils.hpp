#ifndef CUDA_UTILS_HPP
#define CUDA_UTILS_HPP

#include <cuda_runtime_api.h>
#include <memory>

static void CheckCudaError(cudaError_t err, const char *file, int line) {
  if (err != cudaSuccess) {
    printf("[%s@%d]%s in %s:%d\n", cudaGetErrorName(err), err,
           cudaGetErrorString(err), file, line);
    exit(EXIT_FAILURE);
  }
}
#define CHECK_CUDA_ERROR(err) (CheckCudaError(err, __FILE__, __LINE__))

namespace cuda {

constexpr u_int8_t ToDevice = 0;
constexpr u_int8_t ToPin = 1; // to pinned memory

/**
 * @brief 自定义内存回收逻辑
 */
struct deleter {
  void operator()(void *p) const { CHECK_CUDA_ERROR(::cudaFree(p)); }
};

template <typename T> using unique_ptr = std::unique_ptr<T, deleter>;

// 数组类型
template <typename T>
typename std::enable_if<std::is_array<T>::value, cuda::unique_ptr<T>>::type
make_unique(const std::size_t n, u_int8_t flag = 0) {
  // e.g typename std::remove_extent<float[]>::type -> float;
  // 取得数组中元素的类型
  using U = typename std::remove_extent<T>::type;
  U *p;
  switch (flag) {
  case ToDevice: {
    CHECK_CUDA_ERROR(
        ::cudaMalloc(reinterpret_cast<void **>(&p), sizeof(U) * n));
    break;
  }
  case ToPin: {
    CHECK_CUDA_ERROR(
        ::cudaMallocHost(reinterpret_cast<void **>(&p), sizeof(U) * n));
    break;
  }
  default:
    throw std::runtime_error("make_unique Invalid Flag: Must be 0 or 1");
  }
  return cuda::unique_ptr<T>{p};
}

// 普通类型
template <typename T> cuda::unique_ptr<T> make_unique(u_int8_t flag = 0) {
  T *p;
  switch (flag) {
  case ToDevice: {
    CHECK_CUDA_ERROR(::cudaMalloc(reinterpret_cast<void **>(&p), sizeof(T)));
    break;
  }
  case ToPin: {
    CHECK_CUDA_ERROR(
        ::cudaMallocHost(reinterpret_cast<void **>(&p), sizeof(T)));
    break;
  }
  default:
    throw std::runtime_error("make_unique Invalid Flag: Must be 0 or 1");
  }
  return cuda::unique_ptr<T>{p};
}

} // namespace cuda
#endif // CUDA_UTILS_HPP
