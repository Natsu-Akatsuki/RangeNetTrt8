
// opencv stuff
#include <opencv2/core/core.hpp>
#include <opencv2/viz.hpp>

// c++ stuff
#include <chrono>
#include <iomanip> // for setfill
#include <iostream>
#include <string>

// net stuff
#include <selector.hpp>
namespace cl = rangenet::segmentation;

// standalone lib h
#include "infer.hpp"

// boost
#include <boost/program_options.hpp>
namespace po = boost::program_options;
#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;

typedef std::tuple<u_char, u_char, u_char> color;


#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

typedef pcl::PointXYZI PointType;
constexpr int num_point_dims = 4;
/**
 * @brief Converts a binary file to a PCL point cloud.
 * @param filename
 * @param pointcloud
 * @return
 * @note 仅支持读取点云类型为pcl::PointXYZI的点云
 */
bool readBinFile(const std::string &filename,
                 pcl::PointCloud<PointType> &pointcloud) {
  std::ifstream file(filename, std::ios::in | std::ios::binary);
  if (!file) {
    std::cerr << "Could not open the file " << filename << std::endl;
    return false;
  }
  using namespace std;
  // 获取文件大小：设置输入流的位置，挪到文件末尾和文件头
  file.seekg(0, std::ios::end);
  auto file_size = file.tellg();
  file.seekg(0, std::ios::beg);
  size_t point_num =
      static_cast<unsigned int>(file_size) / sizeof(float) / num_point_dims;
  {
    PointType point;
    for (size_t i = 0; i < point_num; ++i) {
      // 将number of characters存放到某个char buffer中
      file.read((char *)&point.x, 3 * sizeof(float));
      // 需要使用intensity，不能直接使用 4 * sizeof(float)
      file.read((char *)&point.intensity, sizeof(float));
      pointcloud.push_back(point);
    }
  }
  file.close();
  return true;
}
inline void pcl22DVector(pcl::PointCloud<PointType> &pointcloud,
                         std::vector<std::vector<float>> &pointcloud_vector) {
  for (const auto &point : pointcloud) {
    pointcloud_vector.emplace_back(std::initializer_list<float>(
        {point.x, point.y, point.z, point.intensity}));
  }
}
inline void pcl21DVector(pcl::PointCloud<PointType> &pointcloud,
                         std::vector<float> &pointcloud_vector) {
  int num_points = pointcloud.size();
  for (int i = 0; i < num_points; i++) {
    pointcloud_vector[i * num_point_dims + 0] = pointcloud.points[i].x;
    pointcloud_vector[i * num_point_dims + 1] = pointcloud.points[i].y;
    pointcloud_vector[i * num_point_dims + 2] = pointcloud.points[i].z;
    pointcloud_vector[i * num_point_dims + 3] = pointcloud.points[i].intensity;
  }
}
inline void vector2Pcl(std::vector<std::vector<float>> &pointcloud_vector,
                       pcl::PointCloud<PointType> &pointcloud) {
  PointType point;
  for (const auto &point_vector : pointcloud_vector) {
    point.x = point_vector[0];
    point.y = point_vector[1];
    point.z = point_vector[2];
    point.intensity = point_vector[3];
    pointcloud.push_back(point);
  }
}

void inferAPI(const std::string &model_path,
              pcl::PointCloud<PointType> &pointcloud,
              std::vector<int> &labels) {

  // create a network
  std::unique_ptr<cl::Net> net = cl::make_net(model_path);

  // pcl->vector
  int num_points = pointcloud.size();
  std::vector<float> pointcloud_vector;
  pointcloud_vector.resize(num_points * num_point_dims);
  pcl21DVector(pointcloud, pointcloud_vector);

  // predict
  std::vector<std::vector<float>> semantic_scan =
      net->infer(pointcloud_vector, num_points);

  // get (N,5) pointcloud
  net->getLabels(semantic_scan, labels);
}

// int main() {
//   std::string filename =
//       "/home/helios/change_ws/rangenet/src/rangenet_lib/example/000000.bin";
//   std::string model_dir =
//       "/home/helios/change_ws/rangenet/src/rangenet_lib/darknet53/";
//
//   auto pointcloud =
//       pcl::PointCloud<PointType>::Ptr(new pcl::PointCloud<PointType>);
//   readBinFile(filename, *pointcloud);
//
//   auto labels = std::vector<int>(pointcloud->size());
//   inferAPI(model_dir, *pointcloud, labels);
//   return 0;
// }