#include "postprocess.hpp"
#include <opencv2/imgproc/imgproc.hpp>

cv::Mat getGaussianKernel(int rows, int cols, double sigmax, double sigmay)
{
  auto gauss_x = cv::getGaussianKernel(cols, sigmax, CV_32F);
  auto gauss_y = cv::getGaussianKernel(rows, sigmay, CV_32F);
  return gauss_x * gauss_y.t();
}

Postprocess::Postprocess(int kernel_size, float sigma) : device_(torch::kCUDA)
{
  kernel_size_ = kernel_size;
  if (kernel_size_ % 2 == 0)
  {
    throw std::runtime_error("Nearest neighbor kernel must be odd number");
  }

  sigma_ = sigma;
  cv::Mat gaussian_kernel_cv =
    getGaussianKernel(kernel_size_, kernel_size_, sigma_, sigma_);
  at::Tensor gaussian_kernel = torch::from_blob(gaussian_kernel_cv.ptr<float>(),
                                                {kernel_size, kernel_size});

  // 使用反高斯核：距离越大，权重越小
  inv_gaussian_kernel_ = (1 - gaussian_kernel).reshape({1, -1, 1}).to(device_);

#if 0
  std::cout << gaussian_kernel << endl;
#endif
}

/**
 * @brief 得到每个激光点的 label(N, 1)
 * @param range_img_torch: float(H,W,1) 深度图
 * @param pointcloud_torch: float(N,1) 激光点云 (depth/range_arr)
 * @param label_img_torch int(H,W,1) 标签图
 * @param px_torch int(N,1) 输入点云投影后的像素坐标 (pixel_x)
 * @param py_torch int(N,1) 输入点云投影后的像素坐标 (pixel_y)
 * @param device: torch::Device
 * @return
 * @note 其领域点是 img 中的 proj 点
 */
using namespace std;

void Postprocess::postprocessKNN(const float range_img[], float range_arr[],
                                 const float label_img[], const float pxs[],
                                 const float pys[], int point_num,
                                 int pointcloud_labels[])
{
  // transfer data t
  const at::Tensor range_img_torch =
    torch::from_blob((float*)range_img, {1, 1, 64, 2048}).to(device_);
  at::Tensor range_torch =
    torch::from_blob((float*)range_arr, {point_num}).to(device_);
  const at::Tensor label_img_torch =
    torch::from_blob((float*)label_img, {1, 1, 64, 2048})
    .to(device_)
    .toType(torch::kFloat32);
  const at::Tensor px_torch =
    torch::from_blob((float*)pxs, {point_num}).to(device_);
  const at::Tensor pys_torch =
    torch::from_blob((float*)pys, {point_num}).to(device_);
  int img_width = 2048;

  /* step1: get the range of neighbors(based on range image)
   * warp_pointcloud_knn_range[1,1,s*s,H*W] ->
   * unwarp_pointcloud_knn_range[1,1,s*s,N] */
  int pad = int((kernel_size_ - 1) / 2);
  namespace F = torch::nn::functional;
  at::Tensor warp_pointcloud_knn_range = F::unfold(
    range_img_torch,
    F::UnfoldFuncOptions({kernel_size_, kernel_size_}).padding({pad, pad}));
  at::Tensor idx_list = (pys_torch * img_width + px_torch).to(torch::kInt64);
  at::Tensor unwarp_pointcloud_knn_range =
    warp_pointcloud_knn_range.index({Slice(), Slice(), idx_list})
                             .to(torch::kFloat32);
  // make the outliers(e.g range<0)
  unwarp_pointcloud_knn_range.index_put_({unwarp_pointcloud_knn_range < 0},
                                         FLT_MAX);
  int center_coordinate = int((kernel_size_ * kernel_size_ - 1) / 2);
  unwarp_pointcloud_knn_range = unwarp_pointcloud_knn_range.index_put_(
    {Slice(), center_coordinate, Slice()}, range_torch);

  /* step2: calculate the norm2 distance between query point and neighbors point
   * and weight the distance.
   * k2_distances[1,s*s,N]*/
  at::Tensor k2_distances =
    at::abs(unwarp_pointcloud_knn_range - range_torch).to(device_);

  k2_distances = k2_distances * inv_gaussian_kernel_;

  /* step3: calculate the topk neighbor */
  // k2_distances[1,s*s,N]->topk_idx[1, k, N]
  auto topk_idx = std::get<1>(k2_distances.topk(5, 1, false, false));

  /* step4: reap the point neighbors' labels information */
  at::Tensor wrap_pointcloud_label = F::unfold(
    label_img_torch,
    F::UnfoldFuncOptions({kernel_size_, kernel_size_}).padding({pad, pad}));
  at::Tensor unwrap_pointcloud_label =
    wrap_pointcloud_label.index({Slice(), Slice(), idx_list});

  /* step5: get the top k predictions from the knn at each pixel */
  at::Tensor topklabel_idx =
    at::gather(unwrap_pointcloud_label, 1, topk_idx).to(device_);

  // remove the low score point, set it "nclasses+1"
  if (cutoff > 0)
  {
    at::Tensor knn_distances = at::gather(k2_distances, 1, topk_idx);
    at::Tensor knn_invalid_idx = knn_distances > cutoff;
    topklabel_idx.index_put_({knn_invalid_idx}, nclasses).to(device_);
  }

  // argmax onehot has an extra class for objects after cutoff
  // vote_result[1,C+1,N]
  at::Tensor vote_result =
    torch::zeros({1, nclasses + 1, point_num},
                 torch::dtype(torch::kInt32).device(torch::kCUDA));
  at::Tensor votes = torch::ones_like(
    topklabel_idx, torch::dtype(torch::kInt32).device(torch::kCUDA));

  vote_result =
    vote_result.scatter_add_(1, topklabel_idx.to(torch::kInt64), votes);

  // 开始投票 (don't let it choose unlabeled or invalid label)
  // pointcloud_labels_torch[1,N]
  at::Tensor pointcloud_labels_torch =
    vote_result.index({Slice(), Slice(1, -1)}).argmax(1) + 1;

  pointcloud_labels_torch =
    pointcloud_labels_torch.reshape({point_num}).to(torch::kCPU);
  auto knn_argmax_out_ = pointcloud_labels_torch.accessor<long, 1>();

  int count = 0;
  for (int i = 0; i < point_num; i++)
  {
    pointcloud_labels[i] = int(knn_argmax_out_[i]);
    if (pointcloud_labels[i] == int(knn_argmax_out_[i]))
    {
      count++;
    }
  }
}
