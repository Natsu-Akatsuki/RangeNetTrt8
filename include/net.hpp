/* Copyright (c) 2019 Andres Milioto, Cyrill Stachniss, University of Bonn.
 *
 *  This file is part of rangenet_lib, and covered by the provided LICENSE file.
 *
 */
#pragma once
#include <iostream>
#include <limits>
#include <string>
#include <vector>

// opencv
#include "opencv2/core.hpp"

#include "yaml-cpp/yaml.h"
#include <project.hpp>
namespace rangenet {
namespace segmentation {

/**
 * @brief      Class for segmentation network inference.
 */
class Net {
public:
  typedef std::tuple<u_char, u_char, u_char> color;

  Net(const std::string &model_path);
  virtual ~Net(){};

  virtual void doInfer(const pcl::PointCloud<PointType> &pointcloud_pcl, int labels[]) = 0;
  std::vector<cv::Vec3b> getLabels(const std::vector<uint32_t> &semantic_scan);

protected:
  // general
  std::string _model_path; // Where to get model weights and cfg

  // image properties
  std::vector<float> _img_means, _img_stds; // mean and std per channel
  // problem properties
  int32_t _n_classes; // number of classes to differ from
  // sensor properties
  double _fov_up, _fov_down; // field of view up and down in radians

  // config
  YAML::Node data_cfg; // yaml nodes with configuration from training
  YAML::Node arch_cfg; // yaml nodes with configuration from training

  std::vector<int> _lable_map;
  std::map<uint32_t, color> _color_map;
  std::map<uint32_t, color> _argmax_to_rgb; // label->color
};

} // namespace segmentation
} // namespace rangenet
