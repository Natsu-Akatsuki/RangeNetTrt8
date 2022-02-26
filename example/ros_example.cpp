#include "netTensorRT.hpp"
#include "pointcloud_io.h"
#include "ros/ros.h"
#include "std_msgs/String.h"
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <sstream>
namespace cl = rangenet::segmentation;
class SemanticSegment {
public:
  SemanticSegment();
  void
  pointcloudCallback(const sensor_msgs::PointCloud2ConstPtr &pointcloud_msg);

private:
  ros::NodeHandle nh_;
  ros::Publisher pub_;
  ros::Subscriber sub_;
  std::unique_ptr<cl::Net> net_;
};

SemanticSegment::SemanticSegment() : nh_("") {
  pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/label_pointcloud", 1);
  sub_ = nh_.subscribe<sensor_msgs::PointCloud2>(
      "/raw_pointcloud", 1, &SemanticSegment::pointcloudCallback, this);
  std::string model_dir =
      "/home/helios/docker_ws/rangenet/src/rangenet_lib/darknet53/";
  net_ = std::unique_ptr<cl::Net>(new cl::NetTensorRT(model_dir));
};

void SemanticSegment::pointcloudCallback(
    const sensor_msgs::PointCloud2ConstPtr &pointcloud_msg) {

  // step1: convert rosmsg to pcl
  pcl::PointCloud<PointType>::Ptr pointcloud_ros(
      new pcl::PointCloud<PointType>());
  pcl::fromROSMsg(*pointcloud_msg, *pointcloud_ros);

  // step2: infer
  auto labels = std::make_unique<int[]>(pointcloud_ros->size());
  net_->infer(*pointcloud_ros, labels.get());
  pcl::PointCloud<pcl::PointXYZRGB> color_pointcloud;

  // step3: publish pointcloud
  sensor_msgs::PointCloud2 ros_msg;
  dynamic_cast<cl::NetTensorRT *>(net_.get())
      ->paintPointCloud(*pointcloud_ros, color_pointcloud, labels.get());
  pcl::toROSMsg(color_pointcloud, ros_msg);
  ros_msg.header = pointcloud_msg->header;
  pub_.publish(ros_msg);
}

int main(int argc, char **argv) {

  ros::init(argc, argv, "pointcloud_semantic_segmentation");
  SemanticSegment semantic_segment;
  while (ros::ok()) {
    ros::spin();
  }
  return 0;
}