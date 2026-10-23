#include "mppi_critics/spatio_temporal_critic.hpp"

#include "geometry_msgs/msg/point.hpp"

#include <cmath>
#include <algorithm>
#include <limits>
#include <fstream>
#include <utility>
#include <iomanip>

#include <xtensor/xmath.hpp>

namespace mppi::critics
{

void SpatioTemporalCritic::initialize()
{
  auto getParam = parameters_handler_->getParamGetter(name_);
  getParam(tracks_topic_, "tracks_topic", std::string("/dynamic_tracks_state"));

  getParam(power_, "cost_power", 1);
  getParam(weight_, "cost_weight", 20.0f);
  getParam(collision_radius_, "collision_radius", 0.5f);
  getParam(vel_std_gain_, "vel_std_gain", 0.4f);
  getParam(min_ped_speed_, "min_ped_speed", 0.4f);
  getParam(max_vel_std_, "max_vel_std", 0.8f);
  getParam(max_track_age_s_, "max_track_age_s", 0.5f);
  getParam(compensate_age_, "compensate_age", true);
  getParam(marker_period_s_, "marker_period_s", 0.2f);
  getParam(prediction_horizon_s_, "prediction_horizon_s", 3.0f);
  getParam(max_radius_, "max_radius", 1.5f);
  getParam(use_footprint_, "use_footprint", false);
  getParam(half_length_, "footprint_half_length", 0.45f);
  getParam(half_width_, "footprint_half_width", 0.33f);
  getParam(front_factor_, "front_factor", 1.0f);
  getParam(worst_case_, "worst_case", true);

  getParam(cost_mode_, "cost_mode", std::string("both"));
  getParam(ttc_weight_, "ttc_weight", 20.0f);
  getParam(ttc_power_, "ttc_power", 1.0f);
  getParam(ttc_epsilon_, "ttc_epsilon", 0.1f);
  getParam(line_weight_, "line_weight", 0.0f);
  getParam(line_clearance_, "line_clearance", 1.0f);
  getParam(line_max_range_, "line_max_range", 6.0f);
  getParam(line_min_cos_, "line_min_cos", 0.7f);

  use_overlap_ = (cost_mode_ == "overlap" || cost_mode_ == "both");
  use_ttc_ = (cost_mode_ == "inv_ttc" || cost_mode_ == "both");
  if (!use_overlap_ && !use_ttc_) {
    RCLCPP_WARN(
      logger_, "invalid cost_mode '%s', falling back to 'both'", cost_mode_.c_str());
    cost_mode_ = "both";
    use_overlap_ = true;
    use_ttc_ = true;
  }

  getParam(diag_period_calls_, "diag_period_calls", 0);

  getParam(diag_csv_path_, "diag_csv_path", std::string(""));
  if (!diag_csv_path_.empty()) {
    diag_csv_.open(diag_csv_path_, std::ios::out | std::ios::trunc);
    if (diag_csv_.is_open()) {
      diag_csv_ << std::fixed << std::setprecision(3);
      diag_csv_ << "t,tracks,contacts,pairs,free,batch,"
                   "tau_min,tau_max,overlap,ttc,pen_max,cost_max,age,dropped_by_age\n";
      RCLCPP_INFO(logger_, "Diagnostics CSV -> %s", diag_csv_path_.c_str());
    } else {
      RCLCPP_WARN(logger_, "Cannot open CSV %s", diag_csv_path_.c_str());
    }
  }

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
    "max_age=%.2fs compensate_age=%d marker every %.2fs",
    cost_mode_.c_str(), weight_, static_cast<int>(worst_case_),
    ttc_weight_, ttc_power_, ttc_epsilon_, prediction_horizon_s_,
    collision_radius_, tracks_topic_.c_str(), max_track_age_s_,
    static_cast<int>(compensate_age_), marker_period_s_);
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
  if (publish_robot_traj_ && robot_pub_ && publish_marker) {
    publishRobotTrajectory(data, dt, time);
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
    last_age_ = age;
    if (age > static_cast<double>(max_track_age_s_)) {
      ++dropped_by_age_;
      if (diag_csv_.is_open()) {
        diag_csv_ << node->now().seconds() << ',' << tracks.size() << ",0,0," << batch << ','
                  << batch << ",-1,-1,0,0,0,0," << age << ',' << dropped_by_age_ << '\n';
        diag_csv_.flush();
      }
      return;
    }
    if (compensate_age_ && age > 0.0) {
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
  std::vector<float> hit_front(batch, 1.0f);

  double diag_sum_ovl = 0.0, diag_sum_ttc = 0.0;
  size_t diag_n_hit = 0, diag_n_tracks = 0;
  size_t diag_tau_min = kNoHit;
  size_t diag_tau_max = 0;

  std::vector<char> diag_libero(batch, 1);

  for (const auto & tr : tracks) {
    const double speed = std::hypot(tr.vx, tr.vy);
    if (speed < min_ped_speed_) {
      continue;
    }
    if (tr.vel_std > max_vel_std_) {
      continue;
    }
    ++diag_n_tracks;

    std::fill(track_pen.begin(), track_pen.end(), 0.0f);
    std::fill(first_hit.begin(), first_hit.end(), kNoHit);
    std::fill(hit_front.begin(), hit_front.end(), 1.0f);

    for (size_t j = 0; j < time_pred; ++j) {
      const float t = static_cast<float>(j) * dt + age_offset;
      const double px = tr.x + tr.vx * t;
      const double py = tr.y + tr.vy * t;
      double radius = collision_radius_ + tr.pos_std +
        vel_std_gain_ * tr.vel_std * t;
      if (radius > max_radius_) {
        radius = max_radius_;
      }
      const double r2 = radius * radius;

      for (size_t i = 0; i < batch; ++i) {
        const double ddx = traj_x(i, j) - px;
        const double ddy = traj_y(i, j) - py;
        double dist;
        if (use_footprint_) {
          const double cy = std::cos(traj_yaws(i, j));
          const double sy = std::sin(traj_yaws(i, j));
          const double u = std::fabs(cy * ddx + sy * ddy);
          const double v = std::fabs(-sy * ddx + cy * ddy);
          dist = std::hypot(std::max(0.0, u - half_length_), std::max(0.0, v - half_width_));
          if (dist >= radius) {
            continue;
          }
        } else {
          const double d2 = ddx * ddx + ddy * ddy;
          if (d2 >= r2) {
            continue;
          }
          dist = std::sqrt(d2);
        }
        float front = 1.0f;
        if (front_factor_ != 1.0f && (ddx * tr.vx + ddy * tr.vy) > 0.0) {
          front = front_factor_;
        }

        if (first_hit[i] == kNoHit) {
          first_hit[i] = j;
          hit_front[i] = front;
        }

        if (!use_overlap_) {
          continue;
        }
        const float closeness =
          front * static_cast<float>(1.0 - dist / radius);
        if (worst_case_) {
          if (closeness > track_pen[i]) {
            track_pen[i] = closeness;
          }
        } else {
          track_pen[i] += closeness;
        }
      }
    }

    if (line_weight_ > 0.0f) {
      const double ux = tr.vx / speed;
      const double uy = tr.vy / speed;
      const double rx = data.state.pose.pose.position.x;
      const double ry = data.state.pose.pose.position.y;
      const auto & q = data.state.pose.pose.orientation;
      const double ryaw =
        std::atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z));
      const double hx = std::cos(ryaw);
      const double hy = std::sin(ryaw);
      const double ahead = (tr.x - rx) * hx + (tr.y - ry) * hy;
      const double range = std::hypot(tr.x - rx, tr.y - ry);
      const double facing = -(ux * hx + uy * hy);
      if (ahead > 0.0 && range < line_max_range_ && facing > line_min_cos_ && dt > 0.0f) {
        const double closing = speed + std::max(0.0, static_cast<double>(data.state.speed.linear.x));
        const double meet = range / std::max(closing, 0.1);
        const size_t jm = std::min(time - 1, static_cast<size_t>(meet / dt));
        for (size_t i = 0; i < batch; ++i) {
          const double lat = std::fabs((traj_x(i, jm) - tr.x) * uy - (traj_y(i, jm) - tr.y) * ux);
          const double f = std::max(0.0, 1.0 - lat / line_clearance_);
          penalty[i] += line_weight_ * static_cast<float>(f);
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
        diag_libero[i] = 0;
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
          c *= ttc_weight_ * hit_front[i];
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

  const bool want_log =
    (diag_period_calls_ > 0 && (++score_calls_ % diag_period_calls_) == 0);
  const bool want_csv = diag_csv_.is_open();

  if (want_log || want_csv) {
    float pen_max = 0.0f, cost_max = 0.0f;
    for (size_t i = 0; i < batch; ++i) {
      if (penalty[i] > pen_max) {
        pen_max = penalty[i];
      }
      if (data.costs(i) > cost_max) {
        cost_max = data.costs(i);
      }
    }
    size_t free_samples = 0;
    for (size_t i = 0; i < batch; ++i) {
      if (diag_libero[i]) {
        ++free_samples;
      }
    }

    const double tau_min = (diag_tau_min == kNoHit)
      ? -1.0 : static_cast<double>(diag_tau_min) * dt;
    const double tau_max = (diag_tau_min == kNoHit)
      ? -1.0 : static_cast<double>(diag_tau_max) * dt;

    if (want_log) {
      RCLCPP_INFO(
        logger_,
        "[STC] mode=%s tracks=%zu contacts=%zu/%zu free=%zu/%zu "
        "tau=[%.2f..%.2f]s | overlap=%.1f ttc=%.1f | "
        "own penalty max=%.1f of total cost max=%.1f",
        cost_mode_.c_str(), diag_n_tracks, diag_n_hit, batch * diag_n_tracks,
        free_samples, batch, tau_min, tau_max,
        diag_sum_ovl, diag_sum_ttc, pen_max, cost_max);
    }
    if (want_csv) {
      double t_now = 0.0;
      if (auto node = parent_.lock()) {
        t_now = node->now().seconds();
      }
      diag_csv_ << t_now << ',' << diag_n_tracks << ',' << diag_n_hit << ','
                << (batch * diag_n_tracks) << ',' << free_samples << ',' << batch << ','
                << tau_min << ',' << tau_max << ',' << diag_sum_ovl << ','
                << diag_sum_ttc << ',' << pen_max << ',' << cost_max << ','
                << last_age_ << ',' << dropped_by_age_ << '\n';
      diag_csv_.flush();
    }
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

void SpatioTemporalCritic::publishRobotTrajectory(
  const mppi::CriticData & data, float dt, size_t time)
{
  const auto & traj_x = data.trajectories.x;
  const auto & traj_y = data.trajectories.y;
  const size_t batch = traj_x.shape(0);
  if (batch == 0) {
    return;
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
  line.ns = "robot_mean";
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
    double mx = 0.0, my = 0.0;
    for (size_t i = 0; i < batch; ++i) {
      mx += traj_x(i, j);
      my += traj_y(i, j);
    }
    mx /= static_cast<double>(batch);
    my /= static_cast<double>(batch);

    geometry_msgs::msg::Point p;
    p.x = mx;
    p.y = my;
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
