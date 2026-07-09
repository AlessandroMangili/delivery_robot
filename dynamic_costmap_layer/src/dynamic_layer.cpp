#include "dynamic_costmap_layer/dynamic_layer.hpp"

#include <algorithm>

#include "nav2_costmap_2d/cost_values.hpp"
#include "pluginlib/class_list_macros.hpp"

using nav2_costmap_2d::NO_INFORMATION;
using nav2_costmap_2d::LETHAL_OBSTACLE;

namespace dynamic_costmap_layer
{

void DynamicLayer::onInitialize()
{
  auto node = node_.lock();
  if (!node) {
    throw std::runtime_error{"DynamicLayer: impossibile bloccare il node"};
  }

  // parametri (namespace = nome del layer, es. "dynamic_layer.<param>")
  declareParameter("enabled", rclcpp::ParameterValue(true));
  declareParameter("dynamic_cost_topic", rclcpp::ParameterValue(std::string("/dynamic_cost")));
  declareParameter("cost_threshold", rclcpp::ParameterValue(1));

  node->get_parameter(name_ + "." + "enabled", enabled_layer_);
  node->get_parameter(name_ + "." + "dynamic_cost_topic", topic_);
  node->get_parameter(name_ + "." + "cost_threshold", cost_threshold_);

  enabled_ = enabled_layer_;
  current_ = true;

  // QoS TRANSIENT_LOCAL + RELIABLE: combacia col publisher del tracker e
  // riceve subito l'ultimo messaggio latchato all'avvio.
  rclcpp::QoS qos(1);
  qos.reliable().transient_local();

  sub_ = node->create_subscription<nav_msgs::msg::OccupancyGrid>(
    topic_, qos,
    std::bind(&DynamicLayer::gridCallback, this, std::placeholders::_1));

  RCLCPP_INFO(
    node->get_logger(),
    "DynamicLayer inizializzato | topic=%s | soglia=%d", topic_.c_str(), cost_threshold_);
}

void DynamicLayer::gridCallback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg)
{
  std::lock_guard<std::mutex> lock(grid_mutex_);
  last_grid_ = msg;
}

void DynamicLayer::updateBounds(
  double /*robot_x*/, double /*robot_y*/, double /*robot_yaw*/,
  double * min_x, double * min_y, double * max_x, double * max_y)
{
  if (!enabled_) {
    return;
  }
  std::lock_guard<std::mutex> lock(grid_mutex_);
  if (!last_grid_) {
    return;
  }

  // estensione mondo della griglia dinamica -> allarga la finestra da ricalcolare
  const auto & info = last_grid_->info;
  double wx_min = info.origin.position.x;
  double wy_min = info.origin.position.y;
  double wx_max = wx_min + info.width * info.resolution;
  double wy_max = wy_min + info.height * info.resolution;

  *min_x = std::min(*min_x, wx_min);
  *min_y = std::min(*min_y, wy_min);
  *max_x = std::max(*max_x, wx_max);
  *max_y = std::max(*max_y, wy_max);
}

void DynamicLayer::updateCosts(
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

  const auto & info = grid->info;
  const double res = info.resolution;
  const double ox = info.origin.position.x;
  const double oy = info.origin.position.y;

  // NOTA FRAME: si assume che /dynamic_cost sia nello stesso frame della local
  // costmap (odom). Con lo stesso frame non serve TF: le coordinate mondo della
  // griglia dinamica coincidono con quelle del master.
  for (unsigned int gy = 0; gy < info.height; ++gy) {
    for (unsigned int gx = 0; gx < info.width; ++gx) {
      int8_t v = grid->data[gy * info.width + gx];
      if (v < cost_threshold_) {
        continue;  // celle libere/sconosciute (-1) o sotto soglia
      }

      // centro cella -> coordinata mondo
      double wx = ox + (gx + 0.5) * res;
      double wy = oy + (gy + 0.5) * res;

      // mondo -> indice del MASTER corrente (gestisce il rolling nativamente)
      unsigned int mi, mj;
      if (!master_grid.worldToMap(wx, wy, mi, mj)) {
        continue;  // fuori dalla finestra della local costmap
      }
      if (static_cast<int>(mi) < min_i || static_cast<int>(mi) >= max_i ||
        static_cast<int>(mj) < min_j || static_cast<int>(mj) >= max_j)
      {
        continue;
      }

      // 0..100 -> 0..254 conservando il GRADIENTE del cono; 100 -> letale
      unsigned char cost = (v >= 100) ?
        LETHAL_OBSTACLE :
        static_cast<unsigned char>((static_cast<int>(v) * 254) / 100);

      // updateWithMax: non sovrascrive un costo piu' alto ne' cancella i muri
      unsigned char old_cost = master_grid.getCost(mi, mj);
      if (old_cost == NO_INFORMATION || old_cost < cost) {
        master_grid.setCost(mi, mj, cost);
      }
    }
  }
}

}  // namespace dynamic_costmap_layer

PLUGINLIB_EXPORT_CLASS(dynamic_costmap_layer::DynamicLayer, nav2_costmap_2d::Layer)