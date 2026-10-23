#include "mppi_critics/heading_critic.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>

#include "nav2_costmap_2d/cost_values.hpp"

namespace mppi::critics
{

void HeadingCritic::initialize()
{
  auto getParam = parameters_handler_->getParamGetter(name_);
  getParam(rotation_weight_, "cost_weight", 10.0f);
  getParam(forward_weight_, "forward_weight", 20.0f);
  getParam(min_rotation_speed_, "min_rotation_speed", 0.20f);
  getParam(horizon_s_, "horizon", 3.0f);
  getParam(align_tolerance_deg_, "align_tolerance_deg", 15.0f);
  getParam(activation_angle_deg_, "activation_angle_deg", 30.0f);
  getParam(deactivation_angle_deg_, "deactivation_angle_deg", 10.0f);
  getParam(heading_lookahead_, "heading_lookahead", 1.0f);
  getParam(goal_distance_, "goal_distance", 0.5f);
  getParam(side_choice_, "side_choice", true);
  getParam(check_distance_, "check_distance", 2.0f);
  getParam(min_distance_, "min_distance", 0.3f);
  getParam(corridor_half_width_, "corridor_half_width", 0.40f);
  getParam(block_cost_, "block_cost", 254);
  getParam(free_cost_, "free_cost", 128);
  getParam(max_offset_, "max_offset", 2.0f);
  getParam(min_gap_, "min_gap", 0.75f);
  getParam(width_cap_, "width_cap", 2.0f);
  getParam(path_half_length_, "path_half_length", 0.4f);
  getParam(space_weight_, "space_weight", 1.0f);
  getParam(deviation_weight_, "deviation_weight", 1.0f);
  getParam(switch_margin_, "switch_margin", 0.5f);
  getParam(lost_side_time_, "lost_side_time", 1.0f);
  getParam(release_time_, "release_time", 2.0f);
  RCLCPP_INFO(
    logger_,
    "HeadingCritic: on above %.0f deg, off below %.0f deg, forward motion penalised while each "
    "trajectory is more than %.0f deg off (first %.1f s) | side choice %s, score = %.1f * width - "
    "%.1f * offset",
    activation_angle_deg_, deactivation_angle_deg_, align_tolerance_deg_, horizon_s_,
    side_choice_ ? "on" : "off", space_weight_, deviation_weight_);
}

int HeadingCritic::cost(double x, double y) const
{
  unsigned int mx, my;
  if (!costmap_->worldToMap(x, y, mx, my)) {
    return nav2_costmap_2d::NO_INFORMATION;
  }
  return costmap_->getCost(mx, my);
}

void HeadingCritic::findGap(
  double bx, double by, double tx, double ty, double nx, double ny, double side,
  float & offset, float & width) const
{
  offset = -1.0f;
  width = 0.0f;
  const double step = 0.05;
  double start = -1.0;
  for (double d = 0.0; d <= max_offset_ + width_cap_; d += step) {
    bool free = true;
    for (double u : {-static_cast<double>(path_half_length_), 0.0, static_cast<double>(path_half_length_)}) {
      const int c = cost(bx + u * tx + side * d * nx, by + u * ty + side * d * ny);
      if (c >= free_cost_) {
        free = false;
        break;
      }
    }
    if (free) {
      if (start < 0.0) {
        if (d > max_offset_) {
          break;
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

void HeadingCritic::score(mppi::CriticData & data)
{
  if (!enabled_) {
    return;
  }
  const double now = std::chrono::duration<double>(
    std::chrono::steady_clock::now().time_since_epoch()).count();
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
  const bool near_goal =
    std::hypot(path.x(path_size - 1) - rx, path.y(path_size - 1) - ry) < goal_distance_;
  bool have_side_target = false;
  if (near_goal) {
    target = path.yaws(path_size - 1);
  } else {
    if (side_choice_) {
      size_t blocked = path_size;
      double s = 0.0;
      for (size_t i = closest + 1; i < path_size; ++i) {
        s += std::hypot(path.x(i) - path.x(i - 1), path.y(i) - path.y(i - 1));
        if (s > check_distance_) {
          break;
        }
        if (s < min_distance_) {
          continue;
        }
        const size_t ia = i > 0 ? i - 1 : i;
        const size_t ib = std::min(i + 1, path_size - 1);
        double ux = path.x(ib) - path.x(ia);
        double uy = path.y(ib) - path.y(ia);
        const double un = std::hypot(ux, uy);
        if (un < 1e-6) {
          continue;
        }
        ux /= un;
        uy /= un;
        bool hit = false;
        for (double off = -corridor_half_width_; off <= corridor_half_width_ + 1e-6;
          off += 0.10)
        {
          const int c = cost(path.x(i) - uy * off, path.y(i) + ux * off);
          if (c >= block_cost_ && c != nav2_costmap_2d::NO_INFORMATION) {
            hit = true;
            break;
          }
        }
        if (hit) {
          blocked = i;
          break;
        }
      }
      if (blocked < path_size) {
        last_blocked_ = now;
        const size_t a = blocked > 0 ? blocked - 1 : blocked;
        const size_t b = std::min(blocked + 1, path_size - 1);
        double tx = path.x(b) - path.x(a);
        double ty = path.y(b) - path.y(a);
        const double tn = std::hypot(tx, ty);
        if (tn > 1e-6) {
          tx /= tn;
          ty /= tn;
          const double nx = -ty;
          const double ny = tx;
          const double bx = path.x(blocked);
          const double by = path.y(blocked);
          float lo, lw, ro, rw;
          findGap(bx, by, tx, ty, nx, ny, 1.0, lo, lw);
          findGap(bx, by, tx, ty, nx, ny, -1.0, ro, rw);
          const bool lok = lo >= 0.0f;
          const bool rok = ro >= 0.0f;
          const float ls = lok ?
            space_weight_ * std::min(lw, width_cap_) - deviation_weight_ * lo : -1e9f;
          const float rs = rok ?
            space_weight_ * std::min(rw, width_cap_) - deviation_weight_ * ro : -1e9f;
          float chosen = side_;
          const bool current_ok = (side_ > 0.0f && lok) || (side_ < 0.0f && rok);
          if (side_ == 0.0f) {
            chosen = (lok || rok) ? (ls >= rs ? 1.0f : -1.0f) : 0.0f;
            side_lost_since_ = -1.0;
          } else if (current_ok) {
            side_lost_since_ = -1.0;
            const float mine = side_ > 0.0f ? ls : rs;
            const float other = side_ > 0.0f ? rs : ls;
            if (other > mine + switch_margin_) {
              chosen = -side_;
            }
          } else {
            if (side_lost_since_ < 0.0) {
              side_lost_since_ = now;
            }
            if (now - side_lost_since_ > lost_side_time_ && (lok || rok)) {
              chosen = lok ? 1.0f : -1.0f;
              side_lost_since_ = -1.0;
            }
          }
          if (chosen != side_) {
            RCLCPP_INFO(
              logger_,
              "HeadingCritic side %s | left offset %.2f width %.2f score %.2f | right offset %.2f "
              "width %.2f score %.2f",
              chosen > 0.0f ? "left" : (chosen < 0.0f ? "right" : "none"), lo, lw, ls, ro, rw, rs);
            side_ = chosen;
          }
          const bool usable = (side_ > 0.0f && lok) || (side_ < 0.0f && rok);
          if (usable) {
            const float off = side_ > 0.0f ? lo : ro;
            const float wid = side_ > 0.0f ? lw : rw;
            const double lateral = off + std::min(wid, 0.8f) * 0.5;
            const double gx = bx + side_ * lateral * nx;
            const double gy = by + side_ * lateral * ny;
            target = std::atan2(gy - ry, gx - rx);
            have_side_target = true;
          }
        }
      } else if (side_ != 0.0f && now - last_blocked_ > release_time_) {
        RCLCPP_INFO(logger_, "HeadingCritic side released: path free for %.1f s", release_time_);
        side_ = 0.0f;
      }
    }
    if (!have_side_target) {
      size_t k = closest;
      double s = 0.0;
      while (k + 1 < path_size && s < heading_lookahead_) {
        s += std::hypot(path.x(k + 1) - path.x(k), path.y(k + 1) - path.y(k));
        ++k;
      }
      target = std::atan2(path.y(k) - ry, path.x(k) - rx);
    }
  }

  double e = target - yaw;
  e = std::atan2(std::sin(e), std::cos(e));
  const double ed = std::fabs(e) * 180.0 / M_PI;
  const bool was = active_;
  active_ = active_ ? ed > deactivation_angle_deg_ : ed > activation_angle_deg_;
  if (active_ != was) {
    RCLCPP_INFO(logger_, "HeadingCritic %s, heading error %.0f deg", active_ ? "on" : "off", ed);
  }
  if (!active_) {
    return;
  }

  const auto & yaws = data.trajectories.yaws;
  const size_t window = std::min(
    steps, std::max<size_t>(1, static_cast<size_t>(std::lround(horizon_s_ / data.model_dt))));
  const float tol = align_tolerance_deg_ * static_cast<float>(M_PI) / 180.0f;
  const float tgt = static_cast<float>(target);
  for (size_t i = 0; i < batch; ++i) {
    float c = 0.0f;
    for (size_t t = 0; t < window; ++t) {
      float err = tgt - yaws(i, t);
      err = std::atan2(std::sin(err), std::cos(err));
      if (std::fabs(err) <= tol) {
        break;
      }
      const float dir = err > 0.0f ? 1.0f : -1.0f;
      c += forward_weight_ * std::max(0.0f, cvx(i, t)) +
        rotation_weight_ * std::max(0.0f, min_rotation_speed_ - dir * cwz(i, t));
    }
    if (c > 0.0f) {
      data.costs(i) += c * data.model_dt;
    }
  }
}

}

#include <pluginlib/class_list_macros.hpp>
PLUGINLIB_EXPORT_CLASS(mppi::critics::HeadingCritic, mppi::critics::CriticFunction)
