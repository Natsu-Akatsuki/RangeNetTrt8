#include "netTensorRT.hpp"
#include "pointcloud_io.h"
#include "yaml-cpp/yaml.h"
#include <iostream>
#include <string>

void inline initialGPU() {
  cudaSetDevice(0);
  cudaFree(0);
}

int main(int argc, const char *argv[]) {
  // step1: initial the GPU
  initialGPU();

  // step2: get the config parameters
  const std::string file_path = __FILE__;
  const std::string config_path =
      file_path.substr(0, file_path.rfind("/")) + "/infer.yaml";
  YAML::Node config = YAML::LoadFile(config_path);
  const std::string data_path = config["DATA_PATH"].as<std::string>();
  const std::string model_dir = config["MODEL_DIR"].as<std::string>();

  // step3: load the pointcloud
  pcl::PointCloud<PointType>::Ptr pointcloud(new pcl::PointCloud<PointType>);
  std::cout << "loading file: " << data_path << std::endl;
  if (pcl::io::loadPCDFile<PointType>(data_path, *pointcloud) == -1) {
    PCL_ERROR("Couldn't read file \n");
    exit(EXIT_FAILURE);
  }

  // step4: create the engine
  namespace cl = rangenet::segmentation;
  auto net = std::unique_ptr<cl::Net>(new cl::NetTensorRT(model_dir));

  // step5: infer
  auto labels = std::make_unique<int[]>(pointcloud->size());
  net->doInfer(*pointcloud, labels.get());
  return 0;
}
