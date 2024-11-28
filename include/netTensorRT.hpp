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
#include <chrono>
#include <cuda_runtime_api.h>
#include <iomanip>


#include <cuda_utils.hpp>
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
    case Severity::kINTERNAL_ERROR:return "[F] ";
    case Severity::kERROR:return "[E] ";
    case Severity::kWARNING:return "[W] ";
    case Severity::kINFO:return "[I] ";
    case Severity::kVERBOSE:return "[V] ";
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
  NetTensorRT(const std::string &model_path, bool use_pcl_viewer=false);

  ~NetTensorRT();

  /**
   * @brief 获点云类别
   * @param scan
   * @return N
   */
  void doInfer(const pcl::PointCloud<PointType> &pointcloud_pcl, int labels[]);
  int getBufferSize(Dims d, DataType t);
  void deserializeEngine(const std::string &engine_path);
  void serializeEngine(const std::string &onnx_path,
                       const std::string &engine_path);
  void paintPointCloud(const pcl::PointCloud<PointType> &pointcloud,
                       pcl::PointCloud<pcl::PointXYZRGB> &color_pointcloud,
                       int labels[]);
  void prepareBuffer();
  std::vector<void *> device_buffers_;
  std::vector<void *> host_buffers_;
  pcl::PointCloud<pcl::PointXYZRGB> color_pointcloud_;
  cudaStream_t stream_ = 0;
  cudaEvent_t start_, stop_;
private:
  std::unique_ptr<IRuntime> runtime_ = nullptr;
  std::unique_ptr<ICudaEngine> engine_ = nullptr;
  std::unique_ptr<IExecutionContext> context_ = nullptr;
  bool use_pcl_viewer_;
  Logger g_logger_;
};

} // namespace segmentation
} // namespace rangenet
