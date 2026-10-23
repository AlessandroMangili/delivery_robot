#include "mppi_critics/side_choice_critic.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>

#include "nav2_costmap_2d/cost_values.hpp"

namespace mppi::critics
{

void SideChoiceCritic::initialize()
{
  auto getParam = parameters_handler_->getParamGetter(name_);
  getParam(rotation_weight_, "cost_weight", 10.0f);
  getParam(forward_weight_, "forward_weight", 10.0f);
  getParam(min_rotation_speed_, "min_rotation_speed", 0.20f);
  getParam(time_window_, "time_window", 1.0f);
  getParam(check_distance_, "check_distance", 2.0f);
  getParam(min_distance_, "min_distance", 0.5f);
  getParam(block_cost_, "block_cost", 253);
  getParam(free_cost_, "free_cost", 128);
  getParam(max_offset_, "max_offset", 2.0f);
  getParam(min_gap_, "min_gap", 0.7f);
  getParam(width_cap_, "width_cap", 2.0f);
  getParam(path_half_length_, "path_half_length", 0.4f);
  getParam(space_weight_, "space_weight", 1.0f);
  getParam(path_weight_, "path_weight", 1.0f);
  getParam(activation_angle_deg_, "activation_angle_deg", 20.0f);
  getParam(deactivation_angle_deg_, "deactivation_angle_deg", 8.0f);
  getParam(release_time_, "release_time", 2.0f);

  RCLCPP_INFO(
    logger_,
    "SideChoiceCritic: path blocked within %.1f m -> side score = %.1f * free width - %.1f * offset "
    "from path, side kept until the path is free for %.1f s",
    check_distance_, space_weight_, path_weight_, release_time_);
}

int SideChoiceCritic::cost(double x, double y) const
{
  unsigned int mx, my;
  if (!costmap_->worldToMap(x, y, mx, my)) {
    return nav2_costmap_2d::NO_INFORMATION;
  }
  return costmap_->getCost(mx, my);
}

void SideChoiceCritic::findGap(
  double bx, double by, double tx, double ty, double nx, double ny, double side,
  float & offset, float & width) const
{
  offset = -1.0f;
  width = 0.0f;
  const double step = 0.05;
  double start = -1.0;
  for (double d = 0.0; d <= max_offset_ + width_cap_; d += step) {
    bool free = true;
    for (double u : {-path_half_length_, 0.0f, path_half_length_}) {
      const int c = cost(bx + u * tx + side * d * nx, by + u * ty + side * d * ny);
      if (c >= free_cost_) {
        free = false;
        break;
      }
    }
    if (free) {
      if (start < 0.0) {
        if (d > max_offset_) {
          return;
        }
        start = d;
      }
      width = static_cast<float>(d - start + step);
    } else if (start >= 0.0) {
      if (width >= min_gap_) {
        offset = static_cast<float>(start);
        return;
      }
      start = -1.0;
      width = 0.0f;
    }
  }
  if (start >= 0.0 && width >= min_gap_) {
    offset = static_cast<float>(start);
  } else {
    width = 0.0f;
  }
}

void SideChoiceCritic::score(mppi::CriticData & data)
{
  if (!enabled_) {
    return;
  }
  const double now = std::chrono::duration<double>(
    std::chrono::steady_clock::now().time_since_epoch()).count();
  const auto & path = data.path;
  const size_t path_size = path.x.shape(0);
  const double rx = data.state.pose.pose.position.x;
  const double ry = data.state.pose.pose.position.y;
  const auto & q = data.state.pose.pose.orientation;
  const double yaw =
    std::atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z));

  size_t blocked_idx = path_size;
  if (path_size >= 3) {
    size_t closest = 0;
    double closest_dist = 1e9;
    for (size_t i = 0; i < path_size; ++i) {
      const double d = std::hypot(path.x(i) - rx, path.y(i) - ry);
      if (d < closest_dist) {
        closest_dist = d;
        closest = i;
      }
    }
    double s = 0.0;
    for (size_t i = closest + 1; i < path_size; ++i) {
      s += std::hypot(path.x(i) - path.x(i - 1), path.y(i) - path.y(i - 1));
      if (s > check_distance_) {
        break;
      }
      if (s < min_distance_) {
        continue;
      }
      const int c = cost(path.x(i), path.y(i));
      if (c >= block_cost_ && c != nav2_costmap_2d::NO_INFORMATION) {
        blocked_idx = i;
        break;
      }
    }
  }

  float heading_error = 0.0f;
  bool have_target = false;
  if (blocked_idx < path_size) {
    const size_t a = blocked_idx > 0 ? blocked_idx - 1 : blocked_idx;
    const size_t b = std::min(blocked_idx + 1, path_size - 1);
    double tx = path.x(b) - path.x(a);
    double ty = path.y(b) - path.y(a);
    const double tn = std::hypot(tx, ty);
    if (tn > 1e-6) {
      last_blocked_ = now;
      tx /= tn;
      ty /= tn;
      const double nx = -ty;
      const double ny = tx;
      const double bx = path.x(blocked_idx);
      const double by = path.y(blocked_idx);
      float left_offset, left_width, right_offset, right_width;
      findGap(bx, by, tx, ty, nx, ny, 1.0, left_offset, left_width);
      findGap(bx, by, tx, ty, nx, ny, -1.0, right_offset, right_width);
      const bool left_ok = left_offset >= 0.0f;
      const bool right_ok = right_offset >= 0.0f;
      const bool kept = (side_ > 0.0f && left_ok) || (side_ < 0.0f && right_ok);
      if (!kept) {
        float chosen = 0.0f;
        if (left_ok && right_ok) {
          const float left_score =
            space_weight_ * std::min(left_width, width_cap_) - path_weight_ * left_offset;
          const float right_score =
            space_weight_ * std::min(right_width, width_cap_) - path_weight_ * right_offset;
          chosen = left_score > right_score ? 1.0f : -1.0f;
        } else if (left_ok) {
          chosen = 1.0f;
        } else if (right_ok) {
          chosen = -1.0f;
        }
        if (chosen != side_) {
          RCLCPP_INFO(
            logger_,
            "SideChoiceCritic side %s | left offset %.2f width %.2f | right offset %.2f width %.2f",
            chosen > 0.0f ? "left" : (chosen < 0.0f ? "right" : "none"),
            left_offset, left_width, right_offset, right_width);
        }
        side_ = chosen;
      }
      if (side_ != 0.0f) {
        const float off = side_ > 0.0f ? left_offset : right_offset;
        const float wid = side_ > 0.0f ? left_width : right_width;
        const double lateral = off + std::min(wid, 0.6f) * 0.5;
        const double gx = bx + side_ * lateral * nx;
        const double gy = by + side_ * lateral * ny;
        double e = std::atan2(gy - ry, gx - rx) - yaw;
        e = std::atan2(std::sin(e), std::cos(e));
        heading_error = static_cast<float>(e * 180.0 / M_PI);
        have_target = true;
      }
    }
  } else if (side_ != 0.0f && now - last_blocked_ > release_time_) {
    RCLCPP_INFO(logger_, "SideChoiceCritic released: path free for %.1f s", release_time_);
    side_ = 0.0f;
  }

  if (!have_target) {
    steering_ = false;
    return;
  }
  const float threshold = steering_ ? deactivation_angle_deg_ : activation_angle_deg_;
  steering_ = std::fabs(heading_error) > threshold;
  if (!steering_) {
    return;
  }

  const float direction = heading_error > 0.0f ? 1.0f : -1.0f;
  const auto & cvx = data.state.cvx;
  const auto & cwz = data.state.cwz;
  const size_t batch = cvx.shape(0);
  const size_t steps = cvx.shape(1);
  if (batch == 0 || steps == 0 || data.model_dt <= 0.0f) {
    return;
  }
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
PLUGINLIB_EXPORT_CLASS(mppi::critics::SideChoiceCritic, mppi::critics::CriticFunction)
