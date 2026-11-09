#include "spatiotemporal_critic/spatiotemporal_critic.hpp"

#include "geometry_msgs/msg/point.hpp"

#include <cmath>
#include <algorithm>
#include <limits>
#include <cstdio>
#include <utility>

#include <xtensor/xmath.hpp>

namespace mppi::critics
{

void SpatioTemporalCritic::initialize()
{
  auto getParam = parameters_handler_->getParamGetter(name_);
  getParam(enabled_, "enabled", true);
  getParam(tracks_topic_, "tracks_topic", std::string("/dynamic_tracks_state"));

  getParam(power_, "cost_power", 1);
  getParam(weight_, "cost_weight", 20.0f);
  getParam(collision_radius_, "collision_radius", 0.5f);
  getParam(vel_std_gain_, "vel_std_gain", 0.4f);
  getParam(min_ped_speed_, "min_ped_speed", 0.4f);
  getParam(max_vel_std_, "max_vel_std", 0.8f);
  getParam(max_track_age_s_, "max_track_age_s", 0.5f);
  getParam(rear_filter_, "rear_filter", false);
  getParam(rear_use_prediction_, "rear_use_prediction", false);
  getParam(rear_margin_m_, "rear_margin_m", 0.3f);
  getParam(rear_keep_close_, "rear_keep_close", true);
  getParam(rear_keep_radius_m_, "rear_keep_radius_m", 1.0f);
  getParam(rear_on_reverse_, "rear_on_reverse", true);
  getParam(prediction_horizon_s_, "prediction_horizon_s", 3.0f);
  getParam(max_radius_, "max_radius", 1.5f);
  getParam(worst_case_, "worst_case", true);

  getParam(cost_mode_, "cost_mode", std::string("both"));
  getParam(ttc_weight_, "ttc_weight", 20.0f);
  getParam(ttc_power_, "ttc_power", 1.0f);
  getParam(ttc_epsilon_, "ttc_epsilon", 0.1f);

  use_overlap_ = (cost_mode_ == "overlap" || cost_mode_ == "both");
  use_ttc_ = (cost_mode_ == "inv_ttc" || cost_mode_ == "both");
  if (!use_overlap_ && !use_ttc_) {
    RCLCPP_WARN(
      logger_, "Invalid cost_mode '%s': falling back to 'both'.", cost_mode_.c_str());
    cost_mode_ = "both";
    use_overlap_ = true;
    use_ttc_ = true;
  }

  getParam(diag_period_calls_, "diag_period_calls", 0);

  getParam(publish_predictions_, "publish_predictions", true);
  getParam(predictions_topic_, "predictions_topic",
    std::string("/spatiotemporal_critic/predictions"));
  getParam(publish_robot_traj_, "publish_robot_trajectory", true);
  getParam(robot_traj_topic_, "robot_trajectory_topic",
    std::string("/spatiotemporal_critic/robot_trajectory"));
  getParam(world_frame_, "world_frame", std::string("odom"));

  auto node = parent_.lock();
  tracks_sub_ = node->create_subscription<dynamic_tracker_msgs::msg::TrackArray>(
    tracks_topic_, rclcpp::QoS(10),
    std::bind(&SpatioTemporalCritic::tracksCallback, this, std::placeholders::_1));

  if (publish_predictions_) {
    pred_pub_ = node->create_publisher<visualization_msgs::msg::MarkerArray>(
      predictions_topic_, rclcpp::QoS(1));
  }
  if (publish_robot_traj_) {
    robot_pub_ = node->create_publisher<visualization_msgs::msg::MarkerArray>(
      robot_traj_topic_, rclcpp::QoS(1));
  }

  RCLCPP_INFO(
    logger_,
    "SpatioTemporalCritic: mode=%s | overlap w=%.1f (worst_case=%d) | "
    "ttc w=%.1f pow=%.2f eps=%.2f | horizon=%.1fs radius=%.2f | tracks_topic=%s | "
    "rear_filter=%d (margin %.2f m, keep_close=%d %.2f m, on_reverse=%d)",
    cost_mode_.c_str(), weight_, static_cast<int>(worst_case_),
    ttc_weight_, ttc_power_, ttc_epsilon_, prediction_horizon_s_,
    collision_radius_, tracks_topic_.c_str(),
    static_cast<int>(rear_filter_), rear_margin_m_, static_cast<int>(rear_keep_close_),
    rear_keep_radius_m_, static_cast<int>(rear_on_reverse_));
}

bool SpatioTemporalCritic::isConsidered(
  double px, double py, double t, const RobotRef & r) const
{
  if (!rear_filter_) {
    return true;
  }
  const double dx = px - (r.x + r.v * t * r.cyaw);
  const double dy = py - (r.y + r.v * t * r.syaw);
  if (rear_keep_close_ &&
    dx * dx + dy * dy < static_cast<double>(rear_keep_radius_m_) * rear_keep_radius_m_)
  {
    return true;
  }
  return dx * r.cyaw + dy * r.syaw > -static_cast<double>(rear_margin_m_);
}

void SpatioTemporalCritic::tracksCallback(
  const dynamic_tracker_msgs::msg::TrackArray::SharedPtr msg)
{
  std::lock_guard<std::mutex> lock(tracks_mutex_);
  tracks_.clear();
  tracks_.reserve(msg->tracks.size());
  for (const auto & t : msg->tracks) {
    tracks_.push_back(TrackSnapshot{t.x, t.y, t.vx, t.vy, t.pos_std, t.vel_std});
  }
  tracks_stamp_ = msg->header.stamp;
  tracks_frame_ = msg->header.frame_id;
}

void SpatioTemporalCritic::score(mppi::CriticData & data)
{
  if (!enabled_) {
    return;
  }

  const auto & traj_x = data.trajectories.x;
  const auto & traj_y = data.trajectories.y;
  const float dt = data.model_dt;
  const size_t batch = traj_x.shape(0);
  const size_t time = traj_x.shape(1);

  if (batch == 0 || time == 0) {
    return;
  }
  if (traj_y.shape(0) != batch || traj_y.shape(1) != time) {
    return;
  }
  if (data.costs.shape(0) != batch) {
    return;
  }

  const bool pub_robot = publish_robot_traj_ && robot_pub_;

  std::vector<TrackSnapshot> tracks;
  builtin_interfaces::msg::Time tracks_stamp;
  {
    std::lock_guard<std::mutex> lock(tracks_mutex_);
    tracks = tracks_;
    tracks_stamp = tracks_stamp_;
  }
  if (tracks.empty()) {
    if (pub_robot) {
      publishRobotTrajectory(data, dt, time);
    }
    return;
  }

  if (auto node = parent_.lock()) {
    const double age = (node->now() - rclcpp::Time(tracks_stamp)).seconds();
    if (age > static_cast<double>(max_track_age_s_)) {
      if (pub_robot) {
        publishRobotTrajectory(data, dt, time);
      }
      return;
    }
  }

  std::vector<float> penalty(batch, 0.0f);

  size_t time_pred = time;
  if (dt > 0.0f) {
    const size_t h = static_cast<size_t>(prediction_horizon_s_ / dt);
    time_pred = std::min(time, std::max<size_t>(1, h));
  }

  const size_t kNoHit = std::numeric_limits<size_t>::max();
  std::vector<float> track_pen(batch, 0.0f);
  std::vector<size_t> first_hit(batch, kNoHit);

  double diag_sum_ovl = 0.0, diag_sum_ttc = 0.0;
  size_t diag_n_hit = 0, diag_n_tracks = 0;
  size_t diag_tau_min = kNoHit;
  size_t diag_tau_max = 0;
  std::vector<char> diag_free(batch, 1);

  const auto & rp = data.state.pose.pose;
  const double ryaw = std::atan2(
    2.0 * (rp.orientation.w * rp.orientation.z + rp.orientation.x * rp.orientation.y),
    1.0 - 2.0 * (rp.orientation.y * rp.orientation.y + rp.orientation.z * rp.orientation.z));
  const RobotRef robot{rp.position.x, rp.position.y, std::cos(ryaw), std::sin(ryaw),
    data.state.speed.linear.x};
  size_t diag_n_behind = 0;

  std::vector<char> reversing(batch, 0);
  size_t diag_n_reversing = 0;
  if (rear_filter_ && rear_on_reverse_) {
    for (size_t i = 0; i < batch; ++i) {
      for (size_t j = 0; j < time_pred; ++j) {
        const double fwd = (traj_x(i, j) - robot.x) * robot.cyaw +
          (traj_y(i, j) - robot.y) * robot.syaw;
        if (fwd < -0.05) {
          reversing[i] = 1;
          ++diag_n_reversing;
          break;
        }
      }
    }
  }
  const bool any_reversing = diag_n_reversing > 0;

  for (const auto & tr : tracks) {
    const double speed = std::hypot(tr.vx, tr.vy);
    if (speed < min_ped_speed_) {
      continue;
    }
    if (tr.vel_std > max_vel_std_) {
      continue;
    }
    const bool in_front_now = isConsidered(tr.x, tr.y, 0.0, robot);
    bool never_in_front = !in_front_now;
    if (rear_use_prediction_) {
      for (size_t j = 0; j < time_pred && never_in_front; ++j) {
        const float t = static_cast<float>(j) * dt;
        never_in_front = !isConsidered(tr.x + tr.vx * t, tr.y + tr.vy * t, t, robot);
      }
    }
    if (never_in_front && !any_reversing) {
      ++diag_n_behind;
      continue;
    }
    ++diag_n_tracks;

    std::fill(track_pen.begin(), track_pen.end(), 0.0f);
    std::fill(first_hit.begin(), first_hit.end(), kNoHit);

    for (size_t j = 0; j < time_pred; ++j) {
      const float t = static_cast<float>(j) * dt;
      const double px = tr.x + tr.vx * t;
      const double py = tr.y + tr.vy * t;
      const bool in_front = rear_use_prediction_ ? isConsidered(px, py, t, robot) : in_front_now;
      if (!in_front && !any_reversing) {
        continue;
      }
      double radius = collision_radius_ + tr.pos_std +
        vel_std_gain_ * tr.vel_std * t;
      if (radius > max_radius_) {
        radius = max_radius_;
      }
      const double r2 = radius * radius;

      for (size_t i = 0; i < batch; ++i) {
        if (!in_front && !reversing[i]) {
          continue;
        }
        const double ddx = traj_x(i, j) - px;
        const double ddy = traj_y(i, j) - py;
        const double d2 = ddx * ddx + ddy * ddy;
        if (d2 >= r2) {
          continue;
        }

        if (first_hit[i] == kNoHit) {
          first_hit[i] = j;
        }

        if (!use_overlap_) {
          continue;
        }
        const float closeness =
          static_cast<float>(1.0 - std::sqrt(d2) / radius);
        if (worst_case_) {
          if (closeness > track_pen[i]) {
            track_pen[i] = closeness;
          }
        } else {
          track_pen[i] += closeness;
        }
      }
    }

    for (size_t i = 0; i < batch; ++i) {
      if (use_overlap_) {
        const float c = weight_ * track_pen[i];
        penalty[i] += c;
        diag_sum_ovl += c;
      }
      if (first_hit[i] != kNoHit) {
        ++diag_n_hit;
        diag_free[i] = 0;
        if (first_hit[i] < diag_tau_min) {
          diag_tau_min = first_hit[i];
        }
        if (first_hit[i] > diag_tau_max) {
          diag_tau_max = first_hit[i];
        }
        if (use_ttc_) {
          const float tau = static_cast<float>(first_hit[i]) * dt;
          float c = 1.0f / (tau + ttc_epsilon_);
          if (std::fabs(ttc_power_ - 1.0f) > 1e-6f) {
            c = std::pow(c, ttc_power_);
          }
          c *= ttc_weight_;
          penalty[i] += c;
          diag_sum_ttc += c;
        }
      }
    }
  }

  for (size_t i = 0; i < batch; ++i) {
    float p = penalty[i];
    if (power_ > 1) {
      p = std::pow(p, static_cast<float>(power_));
    }
    data.costs(i) += p;
  }

  if (pub_robot) {
    publishRobotTrajectory(data, dt, time);
  }

  const bool want_log =
    (diag_period_calls_ > 0 && (++score_calls_ % diag_period_calls_) == 0);

  if (want_log) {
    float pen_max = 0.0f, cost_max = 0.0f;
    for (size_t i = 0; i < batch; ++i) {
      if (penalty[i] > pen_max) {
        pen_max = penalty[i];
      }
      if (data.costs(i) > cost_max) {
        cost_max = data.costs(i);
      }
    }
    size_t n_free = 0;
    for (size_t i = 0; i < batch; ++i) {
      if (diag_free[i]) {
        ++n_free;
      }
    }
    const double tau_min = (diag_tau_min == kNoHit)
      ? -1.0 : static_cast<double>(diag_tau_min) * dt;
    const double tau_max = (diag_tau_min == kNoHit)
      ? -1.0 : static_cast<double>(diag_tau_max) * dt;

    char line[512];
    std::snprintf(
      line, sizeof(line),
      "[STC] mode=%s tracks=%zu ignored_behind=%zu reversing_samples=%zu "
      "contacts=%zu/%zu free=%zu/%zu "
      "tau=[%.2f..%.2f]s | overlap=%.1f ttc=%.1f | "
      "own penalty max=%.1f, cost max=%.1f",
      cost_mode_.c_str(), diag_n_tracks, diag_n_behind, diag_n_reversing,
      diag_n_hit, batch * diag_n_tracks,
      n_free, batch, tau_min, tau_max,
      diag_sum_ovl, diag_sum_ttc, pen_max, cost_max);
    RCLCPP_INFO(logger_, "%s", line);
  }

  if (publish_predictions_ && pred_pub_) {
    publishPredictions(tracks, dt, time, robot);
  }
}

void SpatioTemporalCritic::publishPredictions(
  const std::vector<TrackSnapshot> & tracks,
  float dt, size_t time, const RobotRef & r)
{
  visualization_msgs::msg::MarkerArray arr;

  visualization_msgs::msg::Marker clear;
  clear.action = visualization_msgs::msg::Marker::DELETEALL;
  arr.markers.push_back(clear);

  int id = 1;
  size_t time_pred = time;
  if (dt > 0.0f) {
    const size_t h = static_cast<size_t>(prediction_horizon_s_ / dt);
    time_pred = std::min(time, std::max<size_t>(1, h));
  }
  const size_t step = std::max<size_t>(1, static_cast<size_t>(0.5f / dt));

  for (const auto & tr : tracks) {
    const double speed = std::hypot(tr.vx, tr.vy);
    if (speed < min_ped_speed_) {
      continue;
    }
    if (tr.vel_std > max_vel_std_) {
      continue;
    }
    if (!rear_use_prediction_ && !isConsidered(tr.x, tr.y, 0.0, r)) {
      continue;
    }
    for (size_t j = 0; j < time_pred; j += step) {
      const float t = static_cast<float>(j) * dt;
      const double px = tr.x + tr.vx * t;
      const double py = tr.y + tr.vy * t;
      if (rear_use_prediction_ && !isConsidered(px, py, t, r)) {
        continue;
      }
      double radius = collision_radius_ + tr.pos_std +
        vel_std_gain_ * tr.vel_std * t;
      if (radius > max_radius_) {
        radius = max_radius_;
      }

      visualization_msgs::msg::Marker m;
      m.header.frame_id = tracks_frame_;
      m.header.stamp = tracks_stamp_;
      m.ns = "ped_prediction";
      m.id = id++;
      m.type = visualization_msgs::msg::Marker::SPHERE;
      m.action = visualization_msgs::msg::Marker::ADD;
      m.frame_locked = true;
      m.lifetime = rclcpp::Duration::from_seconds(0.3);
      m.pose.position.x = px;
      m.pose.position.y = py;
      m.pose.position.z = 0.1;
      m.pose.orientation.w = 1.0;
      m.scale.x = 2.0 * radius;
      m.scale.y = 2.0 * radius;
      m.scale.z = 0.05;
      const float frac = static_cast<float>(j) / static_cast<float>(time_pred);
      m.color.r = frac;
      m.color.g = 1.0f - frac;
      m.color.b = 0.0f;
      m.color.a = 0.35f;
      arr.markers.push_back(m);
    }
  }

  pred_pub_->publish(arr);
}

void SpatioTemporalCritic::publishRobotTrajectory(
  const mppi::CriticData & data, float dt, size_t time)
{
  const auto & traj_x = data.trajectories.x;
  const auto & traj_y = data.trajectories.y;
  const size_t batch = traj_x.shape(0);
  if (batch == 0) {
    return;
  }

  size_t best = 0;
  for (size_t i = 1; i < batch; ++i) {
    if (data.costs(i) < data.costs(best)) {
      best = i;
    }
  }

  size_t time_pred = time;
  if (dt > 0.0f) {
    const size_t h = static_cast<size_t>(prediction_horizon_s_ / dt);
    time_pred = std::min(time, std::max<size_t>(1, h));
  }

  rclcpp::Time stamp;
  auto node = parent_.lock();
  if (node) {
    stamp = node->now();
  }

  visualization_msgs::msg::MarkerArray arr;
  visualization_msgs::msg::Marker clear;
  clear.action = visualization_msgs::msg::Marker::DELETEALL;
  arr.markers.push_back(clear);

  visualization_msgs::msg::Marker line;
  line.header.frame_id = world_frame_;
  line.header.stamp = stamp;
  line.ns = "robot_best_sample";
  line.id = 0;
  line.type = visualization_msgs::msg::Marker::LINE_STRIP;
  line.action = visualization_msgs::msg::Marker::ADD;
  line.frame_locked = true;
  line.lifetime = rclcpp::Duration::from_seconds(0.3);
  line.pose.orientation.w = 1.0;
  line.scale.x = 0.05;
  line.color.r = 0.1f;
  line.color.g = 0.3f;
  line.color.b = 1.0f;
  line.color.a = 0.9f;

  for (size_t j = 0; j < time_pred; ++j) {
    geometry_msgs::msg::Point p;
    p.x = traj_x(best, j);
    p.y = traj_y(best, j);
    p.z = 0.15;
    line.points.push_back(p);
  }

  arr.markers.push_back(line);
  robot_pub_->publish(arr);
}

}

#include <pluginlib/class_list_macros.hpp>
PLUGINLIB_EXPORT_CLASS(
  mppi::critics::SpatioTemporalCritic,
  mppi::critics::CriticFunction)
