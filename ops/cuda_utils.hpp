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
make_unique(const std::size_t n) {
  // e.g typename std::remove_extent<float[]>::type -> float;
  // 取得数组中元素的类型
  using U = typename std::remove_extent<T>::type;
  U *p;
  CHECK_CUDA_ERROR(::cudaMalloc(reinterpret_cast<void **>(&p), sizeof(U) * n));
  return cuda::unique_ptr<T>{p};
}

// 普通类型
template <typename T> cuda::unique_ptr<T> make_unique() {
  T *p;
  CHECK_CUDA_ERROR(::cudaMalloc(reinterpret_cast<void **>(&p), sizeof(T)));
  return cuda::unique_ptr<T>{p};
}

} // namespace cuda
#endif // CUDA_UTILS_HPP
