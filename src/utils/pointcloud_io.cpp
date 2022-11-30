#include "pointcloud_io.h"

/**
 * @brief Converts a binary file to a PCL point cloud.
 * @param std::string filename
 * @param pcl::PointCloud<pcl::PointXYZI> pointcloud
 * @return
 * @note 仅支持读取点云类型为pcl::PointXYZI的点云
 */
bool readBinFile(const std::string &filename,
                 pcl::PointCloud<PointType> &pointcloud) {
  std::ifstream file(filename, std::ios::in | std::ios::binary);
  if (!file) {
    auto info =
        std::string("Could not open the pointcloud bin file: ") + filename;
    throw std::runtime_error(info);
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

bool readBinFile(const std::string &filename, std::vector<float> &pointcloud) {
  std::ifstream file(filename, std::ios::in | std::ios::binary);
  if (!file) {
    auto info =
        std::string("Could not open the pointcloud bin file: ") + filename;
    throw std::runtime_error(info);
  }
  using namespace std;
  // 获取文件大小：设置输入流的位置，挪到文件末尾和文件头
  file.seekg(0, std::ios::end);
  auto file_size = file.tellg();
  file.seekg(0, std::ios::beg);

  size_t point_num =
      static_cast<unsigned int>(file_size) / sizeof(float) / num_point_dims;
  pointcloud.resize(num_point_dims * point_num);
  file.read((char *)&pointcloud[0], num_point_dims * point_num * sizeof(float));
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

void pcl21DArray(pcl::PointCloud<PointType> &pointcloud,
                 float *pointcloud_arr) {
  int num_points = pointcloud.size();
  for (int i = 0; i < num_points; i++) {
    pointcloud_arr[i * num_point_dims + 0] = pointcloud.points[i].x;
    pointcloud_arr[i * num_point_dims + 1] = pointcloud.points[i].y;
    pointcloud_arr[i * num_point_dims + 2] = pointcloud.points[i].z;
    pointcloud_arr[i * num_point_dims + 3] = pointcloud.points[i].intensity;
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

