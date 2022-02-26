/* Copyright (c) 2019 Xieyuanli Chen, Andres Milioto, Cyrill Stachniss,
 * University of Bonn.
 *
 *  This file is part of rangenet_lib, and covered by the provided LICENSE file.
 *
 */
#pragma once

#include "net.hpp"
#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <NvOnnxParser.h>
#include <algorithm>
#include <chrono>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iomanip>
#include <ios>
#include <numeric>
#include <pcl/visualization/cloud_viewer.h>
#include <postprocess.hpp>
#include <project.hpp>
using namespace nvinfer1;
namespace rangenet {
namespace segmentation {

/**
 * @brief: 实例化一个ILogger接口类来捕获TensorRT的日志信息
 */
class Logger : public nvinfer1::ILogger {
public:
  // void log(Severity severity, const char *msg)
  void log(Severity severity, const char *msg) noexcept {
    // 设置日志等级
    if (severity <= Severity::kINFO) {
      timePrefix();
      std::cout << severityPrefix(severity) << std::string(msg) << std::endl;
    }
  }

private:
  static const char *severityPrefix(Severity severity) {
    switch (severity) {
    case Severity::kINTERNAL_ERROR:
      return "[F] ";
    case Severity::kERROR:
      return "[E] ";
    case Severity::kWARNING:
      return "[W] ";
    case Severity::kINFO:
      return "[I] ";
    case Severity::kVERBOSE:
      return "[V] ";
    default:
      // #include <cassert>
      assert(0);
      return "";
    }
  }
  void timePrefix() {
    std::time_t timestamp = std::time(nullptr);
    tm *tm_local = std::localtime(&timestamp);
    std::cout << "[";
    std::cout << std::setw(2) << std::setfill('0') << 1 + tm_local->tm_mon
              << "/";
    std::cout << std::setw(2) << std::setfill('0') << tm_local->tm_mday << "/";
    std::cout << std::setw(4) << std::setfill('0') << 1900 + tm_local->tm_year
              << "-";
    std::cout << std::setw(2) << std::setfill('0') << tm_local->tm_hour << ":";
    std::cout << std::setw(2) << std::setfill('0') << tm_local->tm_min << ":";
    std::cout << std::setw(2) << std::setfill('0') << tm_local->tm_sec << "] ";
  }
};

/**
 * @brief      Class for segmentation network inference with TensorRT.
 */
class NetTensorRT : public Net {
public:
  NetTensorRT(const std::string &model_path);

  ~NetTensorRT();

  template <typename T>
  std::vector<size_t> sort_indexes(const std::vector<T> &v) {

    // initialize original index locations
    std::vector<size_t> idx(v.size());
    std::iota(idx.begin(), idx.end(), 0);

    // sort indexes based on comparing values in v. >: decrease <: increase
    std::sort(idx.begin(), idx.end(),
              [&v](size_t i1, size_t i2) { return v[i1] > v[i2]; });

    return idx;
  }

  /**
   * @brief 获点云类别
   * @param scan
   * @return N
   */
  void infer(const pcl::PointCloud<PointType> &pointcloud_pcl, int labels[]);
  int getBufferSize(Dims d, DataType t);
  void deserializeEngine(const std::string &engine_path);
  void serializeEngine(const std::string &onnx_path,
                       const std::string &engine_path);
  void paintPointCloud(const pcl::PointCloud<PointType> &pointcloud,
                       pcl::PointCloud<pcl::PointXYZRGB> &color_pointcloud,
                       int labels[]);
  void prepareBuffer();
  std::vector<void *> _deviceBuffers;
  cudaStream_t stream_;
  std::vector<void *> _hostBuffers;
  pcl::PointCloud<pcl::PointXYZRGB> color_pointcloud_;

protected:
  ICudaEngine *_engine;
  IExecutionContext *_context;
  Logger _gLogger;

  std::vector<float> proj_xs; // store a copy in original order
  std::vector<float> proj_ys;

  // explicitly set the invalid point for both inputs and outputs
  std::vector<float> invalid_input = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  std::vector<float> invalid_output = {1.0f};
};

} // namespace segmentation
} // namespace rangenet
