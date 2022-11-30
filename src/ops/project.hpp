#ifndef RANGENET_LIB_PROJECT_HPP
#define RANGENET_LIB_PROJECT_HPP

#include "cuda_utils.hpp"
#include "project_kernel.hpp"
#include <cuda_runtime_api.h>
#include <iostream>
#include <cmath>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <pcl/point_cloud.h>
#include <pointcloud_io.h>

constexpr int POINT_DIMS = 4;
constexpr int FEATURE_DIMS = 5;
constexpr int FOV_UP = 3;
constexpr int FOV_DOWN = -25;
constexpr float fov_up = FOV_UP * M_PI / 180;
constexpr float fov_down = FOV_DOWN * M_PI / 180;
constexpr float fov = std::abs(fov_down) + std::abs(fov_up);
constexpr int IMG_H = 64;
constexpr int IMG_W = 2048;

class ProjectGPU {
public:
  ProjectGPU(cudaStream_t &stream);
  ~ProjectGPU();
  void doProject(const pcl::PointCloud<PointType> &pointcloud_pcl,
                 bool is_normalized);

  std::unique_ptr<float[]> range_arr_ = nullptr;

  // CPU
  cuda::unique_pin_ptr<float[]> pointcloud_ = nullptr;
  cuda::unique_pin_ptr<float[]> pxs_ = nullptr;
  cuda::unique_pin_ptr<float[]> pys_ = nullptr;
  cuda::unique_pin_ptr<bool[]> valid_idx_ = nullptr;

  // GPU
  cuda::unique_gpu_ptr<bool[]> valid_idx_device_ = nullptr;
  cuda::unique_gpu_ptr<float[]> range_img_device_ = nullptr;

private:
  cudaStream_t stream_;
  int point_num_;
  int pointcloud_size_;

  // CPU
  cuda::unique_pin_ptr<float[]> range_img_ = nullptr;

  // GPU
  cuda::unique_gpu_ptr<float[]> pointcloud_device_ = nullptr;
  cuda::unique_gpu_ptr<float[]> pxs_device_ = nullptr;
  cuda::unique_gpu_ptr<float[]> pys_device_ = nullptr;
};

void createColorImg(float *range_img, int channel);

#endif // RANGENET_LIB_PROJECT_HPP
