#include "project_kernel.hpp"

#if __CUDACC_VER_MAJOR__ == 11 and (__CUDACC_VER_MINOR__ <= 2)
// See https://docs.nvidia.com/cuda/archive/11.1.0/cuda-c-programming-guide/#cpp11-device-variable
__device__ float means[] = {12.12, 10.88, 0.23, -1.04, 0.21};
__device__ float stds[] = {12.32, 11.47, 6.91, 0.86, 0.16};
#elif __CUDACC_VER_MAJOR__ == 11 and (__CUDACC_VER_MINOR__ >= 3) or __CUDACC_VER_MAJOR__ >= 12
__device__ constexpr float means[] = {12.12, 10.88, 0.23, -1.04, 0.21};
__device__ constexpr float stds[] = {12.32, 11.47, 6.91, 0.86, 0.16};
#endif

// __device__ constexpr float means[] = {0.0, 0, 0.0, 0.0, 0.0};
// __device__ constexpr float stds[] = {1.0, 1.0, 1.0, 1.0, 1.0};
/*
 * clamp x to range [a, b]
 * input: x, a, b
 * output:
 * note: __device__ 在device端调用
 */
__device__ float clamp(float x, float a, float b) { return max(a, min(b, x)); }

/*
 * input: pointcloud
 * output: px, py, range_img
 * note: cuda自带math API https://docs.nvidia.com/cuda/cuda-math-api/index.html
 */
__global__ void project_kernel(const float* pointcloud, int point_num,
                               float* pxs, float* pys, bool* valid_idx,
                               float* range_img)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= point_num)
  {
    return;
  }
  float4 point = ((float4*)pointcloud)[idx];
  float range = sqrtf(point.x * point.x + point.y * point.y + point.z * point.z);

  // get angle
  float yaw = -atan2f(point.y, point.x);
  float pitch = asinf(point.z / range);

  float pixel_x = 0.5 * (yaw / M_PI + 1.0); // in [0.0, 1.0]
  float pixel_y = 1.0 - (pitch + std::abs(fov_down)) / fov; // in [0.0, 1.0]
  pixel_x *= IMG_W; // in [0.0, W]
  pixel_y *= IMG_H; // in [0.0, H]

  // clamp
  pixel_x = clamp(floor(pixel_x), 0, IMG_W - 1.0);
  pixel_y = clamp(floor(pixel_y), 0, IMG_H - 1.0);

  pxs[idx] = pixel_x;
  pys[idx] = pixel_y;

  // note: tensorrt network input shape is (C,H,W)
  int HWoffset = int(pixel_y) * IMG_W + int(pixel_x);
  int Coffset = IMG_H * IMG_W;

  // only keep the most close point
  if (valid_idx[HWoffset])
  {
    if (range_img[0 * Coffset + HWoffset] < ((range - means[0]) / stds[0]))
    {
      return;
    }
  }
  // 保证写入时该地址不会被其他线程操作
  atomicExch(range_img + 0 * Coffset + HWoffset, (range - means[0]) / stds[0]);
  atomicExch(range_img + 1 * Coffset + HWoffset, (point.x - means[1]) / stds[1]);
  atomicExch(range_img + 2 * Coffset + HWoffset, (point.y - means[2]) / stds[2]);
  atomicExch(range_img + 3 * Coffset + HWoffset, (point.z - means[3]) / stds[3]);
  atomicExch(range_img + 4 * Coffset + HWoffset, (point.w - means[4]) / stds[4]);

  // 表明为有效的像素格（已有激光点）
  valid_idx[HWoffset] = true;
}

cudaError_t project_launch(const float* pointcloud_device, int point_num,
                           float* pxs_device, float* pys_device, bool* valid_idx_device,
                           float* range_img_device, cudaStream_t stream = 0)
{
  // 执行核函数
  int threadsPerBlock = 256;
  int blocksPerGrid = (point_num + threadsPerBlock - 1) / threadsPerBlock;
  project_kernel<<<blocksPerGrid, threadsPerBlock, 0>>>(
    pointcloud_device, point_num, pxs_device, pys_device, valid_idx_device,
    range_img_device);
  cudaError_t err = cudaGetLastError();
  return err;
}
