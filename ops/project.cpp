#include <project.hpp>

/**
 * @brief 初始化各种指针
 */
ProjectGPU::ProjectGPU(cudaStream_t &stream) {
  pointcloud_ = nullptr;
  pxs_ = nullptr;
  pys_ = nullptr;
  valid_idx_ = nullptr;

  // GPU端
  range_img_device_ = nullptr;

  stream_ = stream;
};

ProjectGPU::~ProjectGPU() {
  // 释放CPU的内存
  CHECK_CUDA_ERROR(cudaFreeHost(pointcloud_));
  CHECK_CUDA_ERROR(cudaFreeHost(pxs_));
  CHECK_CUDA_ERROR(cudaFreeHost(pys_));
  CHECK_CUDA_ERROR(cudaFreeHost(valid_idx_));
};

/**
 *
 * @param pointcloud_ float[N,4] (x,y,z,r)
 * @param isNormalize bool 是否对强度进行归一化
 */
void ProjectGPU::doProject(const pcl::PointCloud<PointType> &pointcloud_pcl,
                           bool isNormalize = false) {
  point_num_ = pointcloud_pcl.size();
  pointcloud_size_ = point_num_ * sizeof(float) * POINT_DIMS;


  // CPU端
  CHECK_CUDA_ERROR(cudaMallocHost((void **)&pointcloud_, pointcloud_size_));
  CHECK_CUDA_ERROR(cudaMallocHost((void **)&pxs_, point_num_ * sizeof(float)));
  CHECK_CUDA_ERROR(cudaMallocHost((void **)&pys_, point_num_ * sizeof(float)));
  CHECK_CUDA_ERROR(cudaMallocHost((void **)&valid_idx_, IMG_H * IMG_W * sizeof(bool)));

  for (int i = 0; i < point_num_; i++) {
    pointcloud_[i * num_point_dims + 0] = pointcloud_pcl.points[i].x;
    pointcloud_[i * num_point_dims + 1] = pointcloud_pcl.points[i].y;
    pointcloud_[i * num_point_dims + 2] = pointcloud_pcl.points[i].z;
    if (isNormalize){
      pointcloud_[i * num_point_dims + 3] = pointcloud_pcl.points[i].intensity / 255;
    }
    else{
      pointcloud_[i * num_point_dims + 3] = pointcloud_pcl.points[i].intensity;
    }
  }

  // GPU端
  pxs_device_ = cuda::make_unique<float[]>(point_num_);
  pys_device_ = cuda::make_unique<float[]>(point_num_);
  valid_idx_device_ = cuda::make_unique<bool[]>(IMG_H * IMG_W);
  pointcloud_device_ = cuda::make_unique<float[]>(point_num_ * POINT_DIMS);
  range_img_device_ = cuda::make_unique<float[]>(IMG_H * IMG_W * FEATURE_DIMS);

  // CPU->GPU
  CHECK_CUDA_ERROR(cudaMemcpy(pointcloud_device_.get(), pointcloud_, pointcloud_size_,
                          cudaMemcpyHostToDevice));
  // execute kernel function
  project_host(pointcloud_, point_num_, pxs_device_.get(), pys_device_.get(),
               valid_idx_device_.get(), range_img_device_.get(), stream_);
  CHECK_CUDA_ERROR(cudaGetLastError());

  // GPU->CPU
  CHECK_CUDA_ERROR(cudaMemcpy(pxs_, pxs_device_.get(), point_num_ * sizeof(float),
                          cudaMemcpyDeviceToHost));
  CHECK_CUDA_ERROR(cudaMemcpy(pys_, pys_device_.get(), point_num_ * sizeof(float),
                          cudaMemcpyDeviceToHost));
  CHECK_CUDA_ERROR(cudaMemcpy(valid_idx_, valid_idx_device_.get(),
                          IMG_H * IMG_W * sizeof(bool), cudaMemcpyDeviceToHost));
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