#include "netTensorRT.hpp"
#include "pointcloud_io.h"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <filesystem>


class ROS_DEMO : public rclcpp::Node {
public:
    ROS_DEMO();

private:
    void pointcloudCallback(sensor_msgs::msg::PointCloud2::SharedPtr pc_msg);

    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
    std::unique_ptr<rangenet::segmentation::Net> net_;
};

ROS_DEMO::ROS_DEMO() : Node("ros2_demo") {

  std::filesystem::path file_path(__FILE__);
  std::string model_dir = std::string(file_path.parent_path().parent_path() / "model/");
  RCLCPP_INFO(this->get_logger(), "model_dir: %s", model_dir.c_str());

  pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/label_pointcloud", 10);
  sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
    "/points_raw", 10, std::bind(&ROS_DEMO::pointcloudCallback, this, std::placeholders::_1));
  net_ = std::unique_ptr<rangenet::segmentation::Net>(new rangenet::segmentation::NetTensorRT(model_dir, false));
};

void ROS_DEMO::pointcloudCallback(
  const sensor_msgs::msg::PointCloud2::SharedPtr pc_msg) {

  // ROS 消息类型 -> PCL 点云类型
  pcl::PointCloud<PointType>::Ptr pc_ros(new pcl::PointCloud<PointType>());
  pcl::fromROSMsg(*pc_msg, *pc_ros);

  // 预测
  auto labels = std::make_unique<int[]>(pc_ros->size());
  net_->doInfer(*pc_ros, labels.get());
  pcl::PointCloud<pcl::PointXYZRGB> color_pc;

  // 发布点云
  sensor_msgs::msg::PointCloud2 ros_msg;
  dynamic_cast<rangenet::segmentation::NetTensorRT *>(net_.get())->paintPointCloud(*pc_ros, color_pc, labels.get());
  pcl::toROSMsg(color_pc, ros_msg);
  ros_msg.header = pc_msg->header;
  pub_->publish(ros_msg);
}

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto ros_demo = std::make_shared<ROS_DEMO>();
  rclcpp::spin(ros_demo);
  rclcpp::shutdown();
  return 0;
}