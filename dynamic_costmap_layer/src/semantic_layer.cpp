#include "dynamic_costmap_layer/semantic_layer.hpp"

#include <algorithm>
#include <cmath>

#include "nav2_costmap_2d/cost_values.hpp"
#include "pluginlib/class_list_macros.hpp"
#include "tf2_ros/buffer.h"
#include "tf2/time.h"
#include "geometry_msgs/msg/transform_stamped.hpp"

using nav2_costmap_2d::NO_INFORMATION;
using nav2_costmap_2d::LETHAL_OBSTACLE;
using nav2_costmap_2d::INSCRIBED_INFLATED_OBSTACLE;

namespace dynamic_costmap_layer
{

void SemanticLayer::onInitialize()
{
  auto node = node_.lock();
  if (!node) {
    throw std::runtime_error{"SemanticLayer: cannot lock the node"};
  }

  declareParameter("enabled", rclcpp::ParameterValue(true));
  declareParameter("map_topic", rclcpp::ParameterValue(std::string("/ssrl_costmap")));

  node->get_parameter(name_ + "." + "enabled", enabled_layer_);
  node->get_parameter(name_ + "." + "map_topic", topic_);
  declareParameter("clear_footprint", rclcpp::ParameterValue(false));
  declareParameter("footprint_margin", rclcpp::ParameterValue(0.05));
  node->get_parameter(name_ + "." + "clear_footprint", clear_footprint_);
  node->get_parameter(name_ + "." + "footprint_margin", footprint_margin_);

  enabled_ = enabled_layer_;
  current_ = true;

  rclcpp::QoS qos(1);
  qos.reliable().durability_volatile();

  sub_ = node->create_subscription<nav_msgs::msg::OccupancyGrid>(
    topic_, qos,
    std::bind(&SemanticLayer::gridCallback, this, std::placeholders::_1));

  RCLCPP_INFO(
    node->get_logger(),
    "SemanticLayer ready: map %s, placed with the map -> odom transform", topic_.c_str());
}

void SemanticLayer::gridCallback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg)
{
  std::lock_guard<std::mutex> lock(grid_mutex_);
  last_grid_ = msg;
}

void SemanticLayer::updateBounds(
  double robot_x, double robot_y, double robot_yaw,
  double * min_x, double * min_y, double * max_x, double * max_y)
{
  if (!enabled_) {
    return;
  }
  robot_x_ = robot_x;
  robot_y_ = robot_y;
  robot_yaw_ = robot_yaw;
  auto * master = layered_costmap_->getCostmap();
  const double ox = master->getOriginX();
  const double oy = master->getOriginY();
  const double sx = master->getSizeInMetersX();
  const double sy = master->getSizeInMetersY();
  *min_x = std::min(*min_x, ox);
  *min_y = std::min(*min_y, oy);
  *max_x = std::max(*max_x, ox + sx);
  *max_y = std::max(*max_y, oy + sy);
}

void SemanticLayer::updateCosts(
  nav2_costmap_2d::Costmap2D & master_grid,
  int min_i, int min_j, int max_i, int max_j)
{
  if (!enabled_) {
    return;
  }

  nav_msgs::msg::OccupancyGrid::SharedPtr grid;
  {
    std::lock_guard<std::mutex> lock(grid_mutex_);
    grid = last_grid_;
  }
  if (!grid) {
    return;
  }

  const std::string map_frame = grid->header.frame_id;
  const std::string local_frame = layered_costmap_->getGlobalFrameID();

  double tx, ty, cyaw, syaw;
  try {
    geometry_msgs::msg::TransformStamped tf =
      tf_->lookupTransform(map_frame, local_frame, tf2::TimePointZero);
    tx = tf.transform.translation.x;
    ty = tf.transform.translation.y;
    const auto & q = tf.transform.rotation;
    const double yaw = std::atan2(
      2.0 * (q.w * q.z + q.x * q.y),
      1.0 - 2.0 * (q.y * q.y + q.z * q.z));
    cyaw = std::cos(yaw);
    syaw = std::sin(yaw);
  } catch (const std::exception & e) {
    return;
  }

  const auto & info = grid->info;
  const double sres = info.resolution;
  const double sox = info.origin.position.x;
  const double soy = info.origin.position.y;
  const int swidth = static_cast<int>(info.width);
  const int sheight = static_cast<int>(info.height);

  const auto & footprint = layered_costmap_->getFootprint();
  double ip_x0 = 0.0, ip_x1 = 0.0, ip_y0 = 0.0, ip_y1 = 0.0;
  for (const auto & q : footprint) {
    ip_x0 = std::min(ip_x0, q.x);
    ip_x1 = std::max(ip_x1, q.x);
    ip_y0 = std::min(ip_y0, q.y);
    ip_y1 = std::max(ip_y1, q.y);
  }
  ip_x0 -= footprint_margin_;
  ip_x1 += footprint_margin_;
  ip_y0 -= footprint_margin_;
  ip_y1 += footprint_margin_;
  const bool use_footprint = clear_footprint_ && !footprint.empty();
  const double rc = std::cos(robot_yaw_);
  const double rs = std::sin(robot_yaw_);

  for (int j = min_j; j < max_j; ++j) {
    for (int i = min_i; i < max_i; ++i) {
      double wx, wy;
      master_grid.mapToWorld(
        static_cast<unsigned int>(i), static_cast<unsigned int>(j), wx, wy);

      const double mx = tx + cyaw * wx - syaw * wy;
      const double my = ty + syaw * wx + cyaw * wy;

      const int sgx = static_cast<int>((mx - sox) / sres);
      const int sgy = static_cast<int>((my - soy) / sres);
      if (sgx < 0 || sgx >= swidth || sgy < 0 || sgy >= sheight) {
        continue;
      }

      const int8_t v = grid->data[sgy * swidth + sgx];
      if (v <= 0) {
        continue;
      }

      unsigned char cost = (v >= 100) ?
        LETHAL_OBSTACLE :
        static_cast<unsigned char>((static_cast<int>(v) * 254) / 100);
      if (use_footprint && cost == LETHAL_OBSTACLE) {
        const double dx = wx - robot_x_;
        const double dy = wy - robot_y_;
        const double u = rc * dx + rs * dy;
        const double w = -rs * dx + rc * dy;
        if (u >= ip_x0 && u <= ip_x1 && w >= ip_y0 && w <= ip_y1) {
          cost = INSCRIBED_INFLATED_OBSTACLE;
        }
      }

      const unsigned char old_cost = master_grid.getCost(
        static_cast<unsigned int>(i), static_cast<unsigned int>(j));
      if (old_cost == NO_INFORMATION || old_cost < cost) {
        master_grid.setCost(
          static_cast<unsigned int>(i), static_cast<unsigned int>(j), cost);
      }
    }
  }
}

}

PLUGINLIB_EXPORT_CLASS(dynamic_costmap_layer::SemanticLayer, nav2_costmap_2d::Layer)
