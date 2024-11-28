#ifndef CUDA_OPS_PROJECT_KERNEL_H
#define CUDA_OPS_PROJECT_KERNEL_H

#include "cuda_utils.hpp"
#include "project.hpp"
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>

cudaError_t project_launch(const float *pointcloud_device, int point_num,
                           float *pxs_device, float *pys_device, bool *valid_idx_device,
                           float *range_img_device, cudaStream_t stream);
#endif // CUDA_OPS_PROJECT_KERNEL_H
