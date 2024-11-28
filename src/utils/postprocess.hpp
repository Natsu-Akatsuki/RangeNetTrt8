#ifndef RANGENET_LIB_POSTPROCESS_H
#define RANGENET_LIB_POSTPROCESS_H

#include <torch/torch.h>

static constexpr int knn = 5;
static constexpr int search = 5;
static constexpr float cutoff = 1.0;
static constexpr int nclasses = 20;

using namespace torch::indexing;

class Postprocess {
public:
  void postprocessKNN(const float range_img[], float range_arr[],
                      const float label_img[], const float pxs[],
                      const float pys[], int point_num, int pointcloud_labels[]
                      );
  Postprocess(int kernel_size, float sigma);

private:
  at::Tensor inv_gaussian_kernel_;
  int kernel_size_;
  float sigma_;
  torch::Device device_;
};

#endif // RANGENET_LIB_POSTPROCESS_H
