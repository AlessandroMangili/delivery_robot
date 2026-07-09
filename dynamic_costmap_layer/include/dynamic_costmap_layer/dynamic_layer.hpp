#ifndef DYNAMIC_COSTMAP_LAYER__DYNAMIC_LAYER_HPP_
#define DYNAMIC_COSTMAP_LAYER__DYNAMIC_LAYER_HPP_

#include <mutex>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "nav2_costmap_2d/layer.hpp"
#include "nav2_costmap_2d/layered_costmap.hpp"
#include "nav_msgs/msg/occupancy_grid.hpp"

namespace dynamic_costmap_layer
{

// Layer che inietta la griglia /dynamic_cost (OccupancyGrid, in odom) nella
// local costmap ROLLING conservando il GRADIENTE del cono (0..100 -> 0..254).
// Scrive con semantica updateWithMax: non cancella mai muri/ostacoli degli
// altri layer, si somma sopra prendendo il massimo.
class DynamicLayer : public nav2_costmap_2d::Layer
{
public:
  DynamicLayer() = default;

  void onInitialize() override;

  void updateBounds(
    double robot_x, double robot_y, double robot_yaw,
    double * min_x, double * min_y, double * max_x, double * max_y) override;

  void updateCosts(
    nav2_costmap_2d::Costmap2D & master_grid,
    int min_i, int min_j, int max_i, int max_j) override;

  void reset() override {}

  bool isClearable() override {return false;}

private:
  void gridCallback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg);

  rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr sub_;
  nav_msgs::msg::OccupancyGrid::SharedPtr last_grid_;
  std::mutex grid_mutex_;

  std::string topic_;
  int cost_threshold_{1};   // valori /dynamic_cost sotto questa soglia -> ignorati
  bool enabled_layer_{true};
};

}  // namespace dynamic_costmap_layer

#endif  // DYNAMIC_COSTMAP_LAYER__DYNAMIC_LAYER_HPP_
