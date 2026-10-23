#include "mppi_critics/rotate_first_critic.hpp"

#include <algorithm>
#include <cmath>

#include "nav2_costmap_2d/cost_values.hpp"

namespace mppi::critics
{

void RotateFirstCritic::initialize()
{
  auto getParam = parameters_handler_->getParamGetter(name_);
  getParam(rotation_weight_, "cost_weight", 10.0f);
  getParam(forward_weight_, "forward_weight", 20.0f);
  getParam(min_rotation_speed_, "min_rotation_speed", 0.20f);
  getParam(time_window_, "time_window", 1.0f);
  getParam(activation_angle_deg_, "activation_angle_deg", 45.0f);
  getParam(deactivation_angle_deg_, "deactivation_angle_deg", 15.0f);
  getParam(heading_lookahead_, "heading_lookahead", 1.0f);
  getParam(goal_distance_, "goal_distance", 0.5f);
  getParam(block_check_distance_, "block_check_distance", 0.0f);
  getParam(block_cost_threshold_, "block_cost_threshold", 254);

  RCLCPP_INFO(
    logger_,
    "RotateFirstCritic: on above %.0f deg, off below %.0f deg | |w| >= %.2f rad/s and forward v "
    "penalised over the first %.1f s | weights w %.1f v %.1f | block check %.1f m",
    activation_angle_deg_, deactivation_angle_deg_, min_rotation_speed_, time_window_,
    rotation_weight_, forward_weight_, block_check_distance_);
}

void RotateFirstCritic::score(mppi::CriticData & data)
{
  if (!enabled_) {
    return;
  }
  const auto & path = data.path;
  const size_t path_size = path.x.shape(0);
  const auto & cvx = data.state.cvx;
  const auto & cwz = data.state.cwz;
  const size_t batch = cvx.shape(0);
  const size_t steps = cvx.shape(1);
  if (path_size < 2 || batch == 0 || steps == 0 || data.model_dt <= 0.0f) {
    active_ = false;
    return;
  }
  const double rx = data.state.pose.pose.position.x;
  const double ry = data.state.pose.pose.position.y;
  const auto & q = data.state.pose.pose.orientation;
  const double yaw =
    std::atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z));

  size_t closest = 0;
  double closest_dist = 1e9;
  for (size_t i = 0; i < path_size; ++i) {
    const double d = std::hypot(path.x(i) - rx, path.y(i) - ry);
    if (d < closest_dist) {
      closest_dist = d;
      closest = i;
    }
  }

  double target;
  if (std::hypot(path.x(path_size - 1) - rx, path.y(path_size - 1) - ry) < goal_distance_) {
    target = path.yaws(path_size - 1);
  } else {
    if (block_check_distance_ > 0.0f) {
      size_t k = closest;
      double s = 0.0;
      while (k + 1 < path_size && s < block_check_distance_) {
        s += std::hypot(path.x(k + 1) - path.x(k), path.y(k + 1) - path.y(k));
        ++k;
        unsigned int mx, my;
        if (s > 0.5 && costmap_->worldToMap(path.x(k), path.y(k), mx, my)) {
          const unsigned char c = costmap_->getCost(mx, my);
          if (c >= block_cost_threshold_ &&
            c != nav2_costmap_2d::NO_INFORMATION)
          {
            if (active_) {
              RCLCPP_INFO(logger_, "RotateFirstCritic off: path blocked");
            }
            active_ = false;
            return;
          }
        }
      }
    }
    size_t k = closest;
    double s = 0.0;
    while (k + 1 < path_size && s < heading_lookahead_) {
      s += std::hypot(path.x(k + 1) - path.x(k), path.y(k + 1) - path.y(k));
      ++k;
    }
    target = std::atan2(path.y(k) - ry, path.x(k) - rx);
  }

  double error = target - yaw;
  error = std::atan2(std::sin(error), std::cos(error));
  const double error_deg = std::fabs(error) * 180.0 / M_PI;
  const bool was_active = active_;
  active_ = active_ ? error_deg > deactivation_angle_deg_ : error_deg > activation_angle_deg_;
  if (active_ != was_active) {
    RCLCPP_INFO(
      logger_, "RotateFirstCritic %s, heading error %.0f deg", active_ ? "on" : "off", error_deg);
  }
  if (!active_) {
    return;
  }

  const float direction = error > 0.0 ? 1.0f : -1.0f;
  const size_t window = std::min(
    steps, std::max<size_t>(1, static_cast<size_t>(std::lround(time_window_ / data.model_dt))));
  const float inv = 1.0f / static_cast<float>(window);
  for (size_t i = 0; i < batch; ++i) {
    float w = 0.0f;
    float v = 0.0f;
    for (size_t t = 0; t < window; ++t) {
      w += cwz(i, t);
      v += cvx(i, t);
    }
    w *= inv;
    v *= inv;
    const float c = rotation_weight_ * std::max(0.0f, min_rotation_speed_ - direction * w) +
      forward_weight_ * std::max(0.0f, v);
    if (c > 0.0f) {
      data.costs(i) += c;
    }
  }
}

}

#include <pluginlib/class_list_macros.hpp>
PLUGINLIB_EXPORT_CLASS(mppi::critics::RotateFirstCritic, mppi::critics::CriticFunction)
