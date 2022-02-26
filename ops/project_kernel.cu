#include "project_kernel.hpp"

__device__ constexpr float means[] = {12.12, 10.88, 0.23, -1.04, 0.21};
__device__ constexpr float stds[] = {12.32, 11.47, 6.91, 0.86, 0.16};
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
__global__ void project_kernel(const float *pointcloud, int point_num,
                               float *pxs, float *pys, bool *valid_idx,
                               float *range_img) {

  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= point_num) {
    return;
  }

  float x = pointcloud[idx * POINT_DIMS + 0];
  float y = pointcloud[idx * POINT_DIMS + 1];
  float z = pointcloud[idx * POINT_DIMS + 2];
  float intensity = pointcloud[idx * POINT_DIMS + 3];
  float range = sqrtf(x * x + y * y + z * z);

  // get angle
  float yaw = -atan2f(y, x);
  float pitch = asinf(z / range);

  float pixel_x = 0.5 * (yaw / M_PI + 1.0);                 // in [0.0, 1.0]
  float pixel_y = 1.0 - (pitch + std::abs(fov_down)) / fov; // in [0.0, 1.0]
  pixel_x *= IMG_W;                                         // in [0.0, W]
  pixel_y *= IMG_H;                                         // in [0.0, H]

  // clamp
  pixel_x = clamp(floor(pixel_x), 0, IMG_W - 1.0);
  pixel_y = clamp(floor(pixel_y), 0, IMG_H - 1.0);

  pxs[idx] = pixel_x;
  pys[idx] = pixel_y;

  // note: tensorrt network input shape is (C,H,W)
  int HWoffset = int(pixel_y) * IMG_W + int(pixel_x);
  int Coffset = IMG_H * IMG_W;
  // 表明为有效的像素格
  valid_idx[HWoffset] = true;

  // 只存储range最小的激光点的属性
  float temp = range_img[0 * Coffset + HWoffset];
  if (fabsf(temp) <= 1e-6 || range < temp) {
    // 保证写入时该地址不会被其他线程操作 (cuda的write本身是原子操作)
    range_img[0 * Coffset + HWoffset] = (range - means[0]) / stds[0];
    range_img[1 * Coffset + HWoffset] = (x - means[1]) / stds[1];
    range_img[2 * Coffset + HWoffset] = (y - means[2]) / stds[2];
    range_img[3 * Coffset + HWoffset] = (z - means[3]) / stds[3];
    range_img[4 * Coffset + HWoffset] = (intensity - means[4]) / stds[4];
  }
}

void project_host(const float *pointcloud_device, int point_num,
                  float *pxs_device, float *pys_device, bool *valid_idx_device,
                  float *range_img_device, cudaStream_t &stream) {
  // 执行核函数
  int threadsPerBlock = 256;
  int blocksPerGrid = (point_num + threadsPerBlock - 1) / threadsPerBlock;
  printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid,
         threadsPerBlock);
  project_kernel<<<blocksPerGrid, threadsPerBlock, 0>>>(
      pointcloud_device, point_num, pxs_device, pys_device, valid_idx_device,
      range_img_device);
}