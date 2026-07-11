#ifndef DYNAMIC_COSTMAP_LAYER__SEMANTIC_LAYER_HPP_
#define DYNAMIC_COSTMAP_LAYER__SEMANTIC_LAYER_HPP_

#include <mutex>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "nav2_costmap_2d/layer.hpp"
#include "nav2_costmap_2d/layered_costmap.hpp"
#include "nav_msgs/msg/occupancy_grid.hpp"

namespace dynamic_costmap_layer
{

// Layer che inietta la mappa semantica SSRL (/ssrl_costmap, in frame 'map')
// nella local costmap ROLLING (in 'odom'), facendo LUI la trasformazione
// map->odom via TF. Serve perche' lo StaticLayer nativo riceve la mappa ma non
// la applica sulla finestra rolling. Conserva il gradiente 0..100 -> 0..254 e
// scrive con updateWithMax: non cancella mai gli altri layer (scan, cono).
class SemanticLayer : public nav2_costmap_2d::Layer
{
public:
  SemanticLayer() = default;

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
  bool enabled_layer_{true};
};

}  // namespace dynamic_costmap_layer

#endif  // DYNAMIC_COSTMAP_LAYER__SEMANTIC_LAYER_HPP_
