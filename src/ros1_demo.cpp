#include "netTensorRT.hpp"
#include "pointcloud_io.h"
#include "ros/ros.h"
#include <filesystem>
#include <functional>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/PointCloud2.h>

class ROS_DEMO {
public:
  explicit ROS_DEMO(ros::NodeHandle *pnh);

private:
  void pointcloudCallback(const sensor_msgs::PointCloud2::ConstPtr &pc_msg);
  ros::NodeHandle *pnh_;
  ros::Publisher pub_;
  ros::Subscriber sub_;
  std::unique_ptr<rangenet::segmentation::Net> net_;
};

ROS_DEMO::ROS_DEMO(ros::NodeHandle *pnh) : pnh_(pnh) {

  std::filesystem::path file_path(__FILE__);
  std::string model_dir = std::string(file_path.parent_path().parent_path() / "model/");
  ROS_INFO("model_dir: %s", model_dir.c_str());

  sub_ = pnh_->subscribe<sensor_msgs::PointCloud2>("/points_raw", 10, &ROS_DEMO::pointcloudCallback, this);
  pub_ = pnh_->advertise<sensor_msgs::PointCloud2>("/label_pointcloud", 1, true);

  net_ = std::unique_ptr<rangenet::segmentation::Net>(new rangenet::segmentation::NetTensorRT(model_dir, false));
};

void ROS_DEMO::pointcloudCallback(
  const sensor_msgs::PointCloud2::ConstPtr &pc_msg) {

  // ROS 消息类型 -> PCL 点云类型
  pcl::PointCloud<PointType>::Ptr pc_ros(new pcl::PointCloud<PointType>());
  pcl::fromROSMsg(*pc_msg, *pc_ros);

  // 预测
  auto labels = std::make_unique<int[]>(pc_ros->size());
  net_->doInfer(*pc_ros, labels.get());
  pcl::PointCloud<pcl::PointXYZRGB> color_pc;

  // 发布点云
  sensor_msgs::PointCloud2 ros_msg;
  dynamic_cast<rangenet::segmentation::NetTensorRT *>(net_.get())->paintPointCloud(*pc_ros, color_pc, labels.get());
  pcl::toROSMsg(color_pc, ros_msg);
  ros_msg.header = pc_msg->header;
  pub_.publish(ros_msg);
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "ros1_demo");
  ros::NodeHandle pnh("~");
  ROS_DEMO node(&pnh);
  ros::spin();
  return 0;
}
