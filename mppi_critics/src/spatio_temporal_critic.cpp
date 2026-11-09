#include "mppi_critics/spatio_temporal_critic.hpp"

#include <cmath>
#include <algorithm>
#include <limits>
#include <utility>

#include <xtensor/xmath.hpp>

namespace mppi::critics
{

void SpatioTemporalCritic::initialize()
{
  auto getParam = parameters_handler_->getParamGetter(name_);
  getParam(tracks_topic_, "tracks_topic", std::string("/dynamic_tracks_state"));
  getParam(weight_, "cost_weight", 20.0f);
  getParam(collision_radius_, "collision_radius", 0.5f);
  getParam(vel_std_gain_, "vel_std_gain", 0.4f);
  getParam(min_ped_speed_, "min_ped_speed", 0.4f);
  getParam(max_vel_std_, "max_vel_std", 0.8f);
  getParam(max_track_age_s_, "max_track_age_s", 0.5f);
  getParam(marker_period_s_, "marker_period_s", 0.2f);
  getParam(prediction_horizon_s_, "prediction_horizon_s", 3.0f);
  getParam(max_radius_, "max_radius", 1.5f);
  getParam(half_length_, "footprint_half_length", 0.45f);
  getParam(half_width_, "footprint_half_width", 0.33f);
  getParam(ttc_weight_, "ttc_weight", 20.0f);
  getParam(ttc_power_, "ttc_power", 1.0f);
  getParam(ttc_epsilon_, "ttc_epsilon", 0.1f);
  getParam(publish_predictions_, "publish_predictions", true);
  getParam(predictions_topic_, "predictions_topic",
    std::string("/spatiotemporal_critic/predictions"));

  auto node = parent_.lock();
  tracks_sub_ = node->create_subscription<dynamic_tracker_msgs::msg::TrackArray>(
    tracks_topic_, rclcpp::QoS(10),
    std::bind(&SpatioTemporalCritic::tracksCallback, this, std::placeholders::_1));

  if (publish_predictions_) {
    pred_pub_ = node->create_publisher<visualization_msgs::msg::MarkerArray>(
      predictions_topic_, rclcpp::QoS(1));
  }

  RCLCPP_INFO(
    logger_,
    "SpatioTemporalCritic: closeness weight %.1f | time to collision weight %.1f power %.2f eps %.2f | "
    "horizon %.1f s, radius %.2f m | tracks %s, max age %.2f s, markers every %.2f s",
    weight_, ttc_weight_, ttc_power_, ttc_epsilon_, prediction_horizon_s_,
    collision_radius_, tracks_topic_.c_str(), max_track_age_s_, marker_period_s_);
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
  const auto & traj_yaws = data.trajectories.yaws;
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

  bool publish_marker = true;
  if (auto node = parent_.lock()) {
    const double now_s = node->now().seconds();
    publish_marker = last_marker_s_ < 0.0 || now_s - last_marker_s_ >= marker_period_s_ ||
      now_s < last_marker_s_;
    if (publish_marker) {
      last_marker_s_ = now_s;
    }
  }

  std::vector<TrackSnapshot> tracks;
  builtin_interfaces::msg::Time tracks_stamp;
  {
    std::lock_guard<std::mutex> lock(tracks_mutex_);
    tracks = tracks_;
    tracks_stamp = tracks_stamp_;
  }

  if (tracks.empty()) {
    return;
  }

  float age_offset = 0.0f;
  if (auto node = parent_.lock()) {
    const double age = (node->now() - rclcpp::Time(tracks_stamp)).seconds();
    if (age > static_cast<double>(max_track_age_s_)) {
      return;
    }
    if (age > 0.0) {
      age_offset = static_cast<float>(age);
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

  for (const auto & tr : tracks) {
    const double speed = std::hypot(tr.vx, tr.vy);
    if (speed < min_ped_speed_) {
      continue;
    }
    if (tr.vel_std > max_vel_std_) {
      continue;
    }

    std::fill(track_pen.begin(), track_pen.end(), 0.0f);
    std::fill(first_hit.begin(), first_hit.end(), kNoHit);

    for (size_t j = 0; j < time_pred; ++j) {
      const float t = static_cast<float>(j) * dt + age_offset;
      const double px = tr.x + tr.vx * t;
      const double py = tr.y + tr.vy * t;
      double radius = collision_radius_ + tr.pos_std +
        vel_std_gain_ * tr.vel_std * t;
      if (radius > max_radius_) {
        radius = max_radius_;
      }

      for (size_t i = 0; i < batch; ++i) {
        const double ddx = traj_x(i, j) - px;
        const double ddy = traj_y(i, j) - py;
        const double cy = std::cos(traj_yaws(i, j));
        const double sy = std::sin(traj_yaws(i, j));
        const double u = std::fabs(cy * ddx + sy * ddy);
        const double v = std::fabs(-sy * ddx + cy * ddy);
        const double dist = std::hypot(std::max(0.0, u - half_length_), std::max(0.0, v - half_width_));
        if (dist >= radius) {
          continue;
        }

        if (first_hit[i] == kNoHit) {
          first_hit[i] = j;
        }

        const float closeness = static_cast<float>(1.0 - dist / radius);
        if (closeness > track_pen[i]) {
          track_pen[i] = closeness;
        }
      }
    }

    for (size_t i = 0; i < batch; ++i) {
      penalty[i] += weight_ * track_pen[i];
      if (first_hit[i] != kNoHit) {
        const float tau = static_cast<float>(first_hit[i]) * dt;
        float c = 1.0f / (tau + ttc_epsilon_);
        if (std::fabs(ttc_power_ - 1.0f) > 1e-6f) {
          c = std::pow(c, ttc_power_);
        }
        c *= ttc_weight_;
        penalty[i] += c;
      }
    }
  }

  for (size_t i = 0; i < batch; ++i) {
    data.costs(i) += penalty[i];
  }

  if (publish_predictions_ && pred_pub_ && publish_marker) {
    publishPredictions(tracks, dt, time, age_offset);
  }
}

void SpatioTemporalCritic::publishPredictions(
  const std::vector<TrackSnapshot> & tracks,
  float dt, size_t time, float age_offset)
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
    for (size_t j = 0; j < time_pred; j += step) {
      const float t = static_cast<float>(j) * dt + age_offset;
      const double px = tr.x + tr.vx * t;
      const double py = tr.y + tr.vy * t;
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

}

#include <pluginlib/class_list_macros.hpp>
PLUGINLIB_EXPORT_CLASS(
  mppi::critics::SpatioTemporalCritic,
  mppi::critics::CriticFunction)
