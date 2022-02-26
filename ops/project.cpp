#include <project.hpp>

/**
 * @brief 初始化各种指针
 */
ProjectGPU::ProjectGPU(cudaStream_t &stream) { stream_ = stream; };

ProjectGPU::~ProjectGPU() = default;

/**
 *
 * @param pointcloud_ float[N,4] (x,y,z,r)
 * @param isNormalize bool 是否对强度进行归一化
 */
void ProjectGPU::doProject(const pcl::PointCloud<PointType> &pointcloud_pcl,
                           bool isNormalize = false) {
  point_num_ = pointcloud_pcl.size();
  pointcloud_size_ = point_num_ * sizeof(float) * POINT_DIMS;

  pointcloud_ = cuda::make_pin_unique<float[]>(point_num_ * POINT_DIMS);
  pxs_ = cuda::make_pin_unique<float[]>(point_num_);
  pys_ = cuda::make_pin_unique<float[]>(point_num_);
  valid_idx_ = cuda::make_pin_unique<bool[]>(IMG_H * IMG_W);

  auto pointcloud_raw_ptr = pointcloud_.get();
  for (int i = 0; i < point_num_; i++) {
    pointcloud_raw_ptr[i * num_point_dims + 0] = pointcloud_pcl.points[i].x;
    pointcloud_raw_ptr[i * num_point_dims + 1] = pointcloud_pcl.points[i].y;
    pointcloud_raw_ptr[i * num_point_dims + 2] = pointcloud_pcl.points[i].z;
    if (isNormalize) {
      pointcloud_raw_ptr[i * num_point_dims + 3] =
          pointcloud_pcl.points[i].intensity / 255;
    } else {
      pointcloud_raw_ptr[i * num_point_dims + 3] =
          pointcloud_pcl.points[i].intensity;
    }
  }

  // GPU端
  pxs_device_ = cuda::make_gpu_unique<float[]>(point_num_);
  pys_device_ = cuda::make_gpu_unique<float[]>(point_num_);
  valid_idx_device_ = cuda::make_gpu_unique<bool[]>(IMG_H * IMG_W);
  pointcloud_device_ = cuda::make_gpu_unique<float[]>(point_num_ * POINT_DIMS);
  range_img_device_ =
      cuda::make_gpu_unique<float[]>(IMG_H * IMG_W * FEATURE_DIMS);

  // CPU->GPU
  CHECK_CUDA_ERROR(cudaMemcpy(pointcloud_device_.get(), pointcloud_.get(),
                              pointcloud_size_, cudaMemcpyHostToDevice));
  // execute kernel function
  project_host(pointcloud_.get(), point_num_, pxs_device_.get(),
               pys_device_.get(), valid_idx_device_.get(),
               range_img_device_.get(), stream_);
  CHECK_CUDA_ERROR(cudaGetLastError());

  // GPU->CPU
  CHECK_CUDA_ERROR(cudaMemcpy(pxs_.get(), pxs_device_.get(),
                              point_num_ * sizeof(float),
                              cudaMemcpyDeviceToHost));
  CHECK_CUDA_ERROR(cudaMemcpy(pys_.get(), pys_device_.get(),
                              point_num_ * sizeof(float),
                              cudaMemcpyDeviceToHost));
  CHECK_CUDA_ERROR(cudaMemcpy(valid_idx_.get(), valid_idx_device_.get(),
                              IMG_H * IMG_W * sizeof(bool),
                              cudaMemcpyDeviceToHost));
}

#if 0  /* some codeblock*/
// CPU端
CHECK_CUDA_ERROR(cudaMallocHost((void **)&range_img_,
                            IMG_H * IMG_W * FEATURE_DIMS * sizeof(float)));
// GPU端
...
// CPU->GPU
CHECK_CUDA_ERROR(cudaMemcpy(pxs_, pxs_device_, point_num_ * sizeof(float),
                        cudaMemcpyDeviceToHost));
CHECK_CUDA_ERROR(cudaMemcpy(pys_, pys_device_, point_num_ * sizeof(float),
                        cudaMemcpyDeviceToHost));
CHECK_CUDA_ERROR(cudaMemcpy(range_img_, range_img_device_,
                        IMG_H * IMG_W * FEATURE_DIMS * sizeof(float),
                        cudaMemcpyDeviceToHost));
CHECK_CUDA_ERROR(cudaFreeHost(range_img_));
#endif /* some codeblock*/

void createColorImg(float *range_img, int channel) {
  uint8_t img_depth[IMG_H * IMG_W];
  int normalization[] = {100, 100, 100, 100, 1};
  int Coffset = IMG_H * IMG_W;

  for (int pixel_x = 0; pixel_x < IMG_W; pixel_x++) {
    for (int pixel_y = 0; pixel_y < IMG_H; pixel_y++) {
      int HWoffset = int(pixel_y) * IMG_W + int(pixel_x);
      img_depth[HWoffset] = range_img[channel * Coffset + HWoffset] /
                            normalization[channel] * 255;
    }
  }

  cv::Mat color_img = cv::Mat(IMG_H, IMG_W, CV_8UC1, &img_depth);
  // 使用对比度高的jet色系
  cv::applyColorMap(color_img, color_img, cv::COLORMAP_JET);
  cv::imshow("color_img", color_img);
  while (1) {
    if (cv::waitKey(0) == 27)
      break;
  }
}