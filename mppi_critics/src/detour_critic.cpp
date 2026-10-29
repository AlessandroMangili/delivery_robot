#include "mppi_critics/detour_critic.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <queue>

#include "nav2_costmap_2d/cost_values.hpp"

namespace mppi::critics
{

namespace
{
double steadyNow()
{
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
}

void DetourCritic::initialize()
{
  auto getParam = parameters_handler_->getParamGetter(name_);
  getParam(attraction_weight_, "cost_weight", 5.0f);
  getParam(rotation_weight_, "rotation_weight", 10.0f);
  getParam(forward_weight_, "forward_weight", 10.0f);
  getParam(min_rotation_speed_, "min_rotation_speed", 0.25f);
  getParam(horizon_s_, "horizon", 3.0f);
  getParam(align_deg_, "align_tolerance_deg", 20.0f);
  getParam(check_distance_, "check_distance", 2.0f);
  getParam(min_distance_, "min_distance", 0.3f);
  getParam(corridor_half_width_, "corridor_half_width", 0.20f);
  getParam(block_cost_, "block_cost", 254);
  getParam(free_cost_, "free_cost", 128);
  getParam(max_offset_, "max_offset", 1.2f);
  getParam(min_gap_, "min_gap", 0.8f);
  getParam(width_cap_, "width_cap", 2.0f);
  getParam(path_half_length_, "path_half_length", 0.4f);
  getParam(space_weight_, "space_weight", 1.0f);
  getParam(deviation_weight_, "deviation_weight", 1.0f);
  getParam(switch_margin_, "switch_margin", 0.5f);
  getParam(progress_distance_, "progress_distance", 0.15f);
  getParam(stuck_turn_deg_, "stuck_turn_deg", 20.0f);
  getParam(stuck_time_, "stuck_time", 1.2f);
  getParam(near_clearance_, "near_clearance", 0.25f);
  getParam(release_time_, "release_time", 1.5f);
  getParam(refresh_distance_, "refresh_distance", 0.5f);
  getParam(goal_distance_, "goal_distance", 1.0f);
  getParam(half_length_, "footprint_half_length", 0.45f);
  getParam(half_width_, "footprint_half_width", 0.33f);
  getParam(ramp_time_, "ramp_time", 1.0f);
  getParam(ramp_out_time_, "ramp_out_time", 1.5f);
  getParam(side_hold_time_, "side_hold_time", 3.0f);
  getParam(side_lost_time_, "side_lost_time", 1.0f);
  getParam(side_memory_time_, "side_memory_time", 30.0f);
  getParam(side_memory_distance_, "side_memory_distance", 1.0f);
  getParam(score_alpha_, "score_alpha", 0.3f);
  getParam(heading_weight_, "heading_weight", 0.5f);
  getParam(reach_penalty_, "reach_penalty", 1.0f);
  getParam(lateral_margin_, "lateral_margin", 0.15f);
  getParam(waypoint_ahead_, "waypoint_ahead", 0.3f);
  getParam(station_step_, "station_step", 0.2f);
  getParam(obstacle_max_length_, "obstacle_max_length", 2.0f);
  getParam(free_length_, "free_length", 0.3f);
  getParam(bend_weight_, "bend_weight", 0.5f);
  getParam(bend_lookahead_, "bend_lookahead", 1.5f);
  getParam(ignore_movers_, "ignore_moving_tracks", true);
  getParam(mover_speed_, "moving_speed", 0.3f);
  getParam(mover_radius_, "moving_radius", 0.7f);
  getParam(max_track_age_s_, "max_track_age_s", 1.0f);
  getParam(tracks_topic_, "tracks_topic", std::string("/dynamic_tracks_state"));
  getParam(use_routes_, "use_routes", true);
  getParam(route_cost_weight_, "route_cost_weight", 10.0f);
  getParam(route_switch_ratio_, "route_switch_ratio", 0.8f);
  getParam(route_mean_cost_, "route_mean_cost", 128.0f);
  getParam(route_period_, "route_period", 1.0f);
  getParam(route_refresh_distance_, "route_refresh_distance", 0.3f);
  getParam(route_goal_margin_, "route_goal_margin", 0.6f);
  getParam(waypoint_spacing_, "waypoint_spacing", 0.5f);
  getParam(reach_ahead_, "reach_ahead", 0.1f);
  getParam(reach_radius_, "reach_radius", 0.35f);
  getParam(passive_gain_, "passive_gain", 0.5f);
  getParam(pre_wp1_gain_, "pre_wp1_gain", 1.0f);
  getParam(straight_segments_, "straight_segments", false);
  getParam(segment_margin_, "segment_margin", 0.10f);
  getParam(turn_margin_, "turn_margin", 0.05f);
  getParam(align_exit_deg_, "align_exit_deg", -1.0f);
  getParam(passive_align_deg_, "passive_align_deg", 60.0f);
  getParam(pass_margin_, "pass_margin", 0.2f);
  getParam(debug_, "debug_topics", false);

  if (debug_) {
    auto node = parent_.lock();
    debug_pub_ = node->create_publisher<visualization_msgs::msg::MarkerArray>("detour_debug", 10);
    frame_ = costmap_ros_->getGlobalFrameID();
  }
  if (ignore_movers_) {
    auto node = parent_.lock();
    tracks_sub_ = node->create_subscription<dynamic_tracker_msgs::msg::TrackArray>(
      tracks_topic_, rclcpp::QoS(10),
      std::bind(&DetourCritic::tracksCallback, this, std::placeholders::_1));
  }
  RCLCPP_INFO(
    logger_,
    "\033[38;5;208mDetourCritic: on when path blocked within %.1f m and (progress < %.2f m in %.1f s or "
    "clearance < %.2f m or reversing) | ramp in %.1f s out %.1f s | side hold %.1f s | "
    "ignore moving tracks %s | sides from %s\033[0m",
    check_distance_, progress_distance_, stuck_time_, near_clearance_, ramp_time_, ramp_out_time_,
    side_hold_time_, ignore_movers_ ? "yes" : "no",
    use_routes_ ? "left/right routes with waypoints" : "lateral scan");
}

void DetourCritic::tracksCallback(const dynamic_tracker_msgs::msg::TrackArray::SharedPtr msg)
{
  std::vector<Mover> m;
  for (const auto & t : msg->tracks) {
    if (std::hypot(t.vx, t.vy) >= mover_speed_) {
      m.push_back(Mover{t.x, t.y});
    }
  }
  std::lock_guard<std::mutex> lock(tracks_mutex_);
  movers_ = std::move(m);
  movers_time_ = steadyNow();
}

std::vector<DetourCritic::Mover> DetourCritic::currentMovers(double now) const
{
  std::lock_guard<std::mutex> lock(tracks_mutex_);
  if (!ignore_movers_ || now - movers_time_ > max_track_age_s_) {
    return {};
  }
  return movers_;
}

float DetourCritic::level(double now) const
{
  if (active_) {
    return ramp_time_ > 0.0f ?
           static_cast<float>(std::clamp((now - on_time_) / ramp_time_, 0.0, 1.0)) : 1.0f;
  }
  if (releasing_) {
    if (ramp_out_time_ <= 0.0f) {
      return 0.0f;
    }
    const double f = 1.0 - (now - release_start_) / ramp_out_time_;
    return release_from_ * static_cast<float>(std::clamp(f, 0.0, 1.0));
  }
  return 0.0f;
}

void DetourCritic::score(CriticData & data)
{
  if (!enabled_) {
    return;
  }
  const double now = steadyNow();
  const auto & path = data.path;
  const size_t path_size = path.x.shape(0);
  const auto & traj_x = data.trajectories.x;
  const auto & traj_y = data.trajectories.y;
  const auto & traj_yaw = data.trajectories.yaws;
  const auto & cvx = data.state.cvx;
  const auto & cwz = data.state.cwz;
  const size_t batch = cvx.shape(0);
  const size_t steps = cvx.shape(1);
  const double rx = data.state.pose.pose.position.x;
  const double ry = data.state.pose.pose.position.y;
  if (path_size < 2 || batch == 0 || steps == 0 || data.model_dt <= 0.0f) {
    deactivate("no path", now, rx, ry);
    releasing_ = false;
    return;
  }
  const auto & q = data.state.pose.pose.orientation;
  const double yaw =
    std::atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z));
  const double v = data.state.speed.linear.x;

  poses_.push_back({now, rx, ry, yaw});
  bool stuck = false;
  while (poses_.size() > 1 && now - poses_[1][0] >= stuck_time_) {
    poses_.pop_front();
  }
  if (now - poses_.front()[0] >= stuck_time_) {
    const auto & p = poses_.front();
    const double turned = std::fabs(std::atan2(std::sin(yaw - p[3]), std::cos(yaw - p[3])));
    stuck = std::hypot(rx - p[1], ry - p[2]) < progress_distance_ &&
      turned * 180.0 / M_PI < stuck_turn_deg_;
  }

  const bool near_goal =
    std::hypot(path.x(path_size - 1) - rx, path.y(path_size - 1) - ry) < goal_distance_;

  const std::vector<Mover> movers = currentMovers(now);
  size_t blocked = path_size;
  double tx = 0.0, ty = 0.0;
  if (!near_goal) {
    findBlocked(data, rx, ry, movers, blocked, tx, ty);
  }
  const bool is_blocked = blocked < path_size;
  if (is_blocked) {
    last_blocked_ = now;
  }

  bool fresh = false;
  if (!active_) {
    const bool resume = releasing_ && is_blocked;
    bool trigger = false;
    const char * why = "";
    if (!resume && is_blocked) {
      const bool near = nearObstacle(rx, ry, yaw);
      const bool reversing = v < -0.02;
      trigger = stuck || near || reversing;
      why = near ? "close to obstacle" : (reversing ? "reversing" : "no progress");
    }
    if (resume || trigger) {
      const float start = level(now);
      active_ = true;
      on_time_ = now - static_cast<double>(start) * ramp_time_;
      if (resume) {
        RCLCPP_INFO(logger_, "\033[38;5;208mDetourCritic on again while releasing (level %.2f)\033[0m", start);
      } else {
        fresh = true;
        side_ = 0.0f;
        have_scores_ = false;
        have_wp_ = false;
        have_exit_ = false;
        have_block_ = false;
        passive_ = false;
        wps_.clear();
        route_used_.clear();
        route_time_ = -1e9;
        last_target_ = std::numeric_limits<size_t>::max();
        RCLCPP_INFO(logger_, "\033[38;5;208mDetourCritic on (%s)\033[0m", why);
      }
      releasing_ = false;
    }
  } else if (near_goal || (!is_blocked && now - last_blocked_ > release_time_ && passedBlock(rx, ry))) {
    const float from = level(now);
    RCLCPP_INFO(
      logger_, "\033[38;5;208mDetourCritic releasing (%s) from level %.2f\033[0m", near_goal ? "near goal" : "path free",
      from);
    active_ = false;
    releasing_ = true;
    release_start_ = now;
    release_from_ = from;
    if (side_ != 0.0f) {
      memory_side_ = side_;
      memory_time_ = now;
    }
  }

  if (!active_ && !releasing_) {
    return;
  }
  if (releasing_ && level(now) <= 0.0f) {
    releasing_ = false;
    side_ = 0.0f;
    have_wp_ = false;
    have_exit_ = false;
    have_scores_ = false;
    have_block_ = false;
    passive_ = false;
    wps_.clear();
    route_used_.clear();
    RCLCPP_INFO(logger_, "\033[38;5;208mDetourCritic off\033[0m");
    return;
  }

  if (active_ && use_routes_ && (is_blocked || (have_block_ && routeBlockedAhead(rx, ry)))) {
    size_t bi = blocked;
    double btx = tx, bty = ty;
    if (!is_blocked) {
      double best = 1e18;
      for (size_t i = 0; i < path_size; ++i) {
        const double d = std::hypot(path.x(i) - block_x_, path.y(i) - block_y_);
        if (d < best) {
          best = d;
          bi = i;
        }
      }
      btx = block_tx_;
      bty = block_ty_;
    }
    const double nx = -bty;
    const double ny = btx;
    const double bx = path.x(bi);
    const double by = path.y(bi);
    const double length = is_blocked ? obstacleLength(bx, by, btx, bty, nx, ny, movers) : block_len_;
    if (have_block_ && std::hypot(bx - block_x_, by - block_y_) > refresh_distance_) {
      passive_ = false;
    }
    routeSides(data, rx, ry, bi, bx, by, btx, bty, nx, ny, length, now);
    have_block_ = true;
    block_x_ = bx;
    block_y_ = by;
    block_tx_ = btx;
    block_ty_ = bty;
    block_len_ = length;
    const bool lok = route_okl_ && route_ml_ < route_mean_cost_;
    const bool rok = route_okr_ && route_mr_ < route_mean_cost_;
    const float ls = -static_cast<float>(route_cl_);
    const float rs = -static_cast<float>(route_cr_);
    if (!have_scores_) {
      left_score_ = ls;
      right_score_ = rs;
      have_scores_ = true;
    } else {
      if (lok) {
        left_score_ += score_alpha_ * (ls - left_score_);
      }
      if (rok) {
        right_score_ += score_alpha_ * (rs - right_score_);
      }
    }
    float chosen = side_;
    const char * reason = "";
    if (side_ == 0.0f) {
      const bool remembered = memory_side_ != 0.0f && now - memory_time_ < side_memory_time_ &&
        std::hypot(bx - memory_x_, by - memory_y_) < side_memory_distance_ &&
        (memory_side_ > 0.0f ? lok : rok);
      if (fresh && remembered) {
        chosen = memory_side_;
        reason = "remembered";
      } else if (lok || rok) {
        chosen = (lok && (!rok || left_score_ >= right_score_)) ? 1.0f : -1.0f;
        reason = "cheaper route";
      }
    } else {
      const bool mine_ok = side_ > 0.0f ? lok : rok;
      const bool other_ok = side_ > 0.0f ? rok : lok;
      if (mine_ok) {
        side_ok_time_ = now;
      }
      const double mine = side_ > 0.0f ? route_cl_ : route_cr_;
      const double other = side_ > 0.0f ? route_cr_ : route_cl_;
      if (!mine_ok && !other_ok && now - side_ok_time_ > side_lost_time_) {
        chosen = 0.0f;
        reason = "no route";
      } else if (other_ok && now - side_ok_time_ > side_lost_time_) {
        chosen = -side_;
        reason = "route lost";
      } else if (other_ok && mine_ok && now - side_time_ > side_hold_time_ &&
        other < route_switch_ratio_ * mine)
      {
        chosen = -side_;
        reason = "clearly cheaper";
      }
    }
    if (chosen != side_) {
      RCLCPP_INFO(
        logger_,
        "\033[38;5;208mDetourCritic side %s (%s) | obstacle %.2f m | left %s cost %.1f mean %.0f | right %s cost %.1f mean %.0f\033[0m",
        chosen > 0.0f ? "left" : (chosen < 0.0f ? "right" : "none"), reason, length,
        route_okl_ ? "ok" : "no", route_okl_ ? route_cl_ : -1.0, route_ml_,
        route_okr_ ? "ok" : "no", route_okr_ ? route_cr_ : -1.0, route_mr_);
      side_ = chosen;
      side_time_ = now;
      side_ok_time_ = now;
      have_wp_ = false;
      passive_ = false;
    }
    if (side_ != 0.0f) {
      memory_x_ = bx;
      memory_y_ = by;
      const bool ok = side_ > 0.0f ? lok : rok;
      if (ok) {
        buildWaypoints(side_ > 0.0f ? route_l_ : route_r_);
        have_wp_ = !wps_.empty();
      }
    } else {
      have_wp_ = false;
      wps_.clear();
    }
    if (debug_ && !have_wp_) {
      publishDebug(now, rx, ry, false, 0, true, 0.0, 0.0, level(now));
    }
  } else if (active_ && is_blocked && !use_routes_) {
    const double nx = -ty;
    const double ny = tx;
    const double bx = path.x(blocked);
    const double by = path.y(blocked);
    const double length = obstacleLength(bx, by, tx, ty, nx, ny, movers);
    float lo, lw, ro, rw;
    const bool lok = sideGap(bx, by, tx, ty, nx, ny, 1.0, length, lo, lw);
    const bool rok = sideGap(bx, by, tx, ty, nx, ny, -1.0, length, ro, rw);
    const double along = 0.5 * length + waypoint_ahead_;
    const double ax = bx + along * tx;
    const double ay = by + along * ty;
    const double l_lat = lo + std::min(lateral_margin_, 0.5f * lw);
    const double r_lat = ro + std::min(lateral_margin_, 0.5f * rw);
    const double lwx = ax + l_lat * nx, lwy = ay + l_lat * ny;
    const double rwx = ax - r_lat * nx, rwy = ay - r_lat * ny;
    const double facing = std::sin(yaw - std::atan2(ty, tx));
    const double bend = pathBend(data, blocked, length, nx, ny);
    float ls = space_weight_ * std::min(lw, width_cap_) - deviation_weight_ * lo +
      heading_weight_ * static_cast<float>(facing) + bend_weight_ * static_cast<float>(bend);
    float rs = space_weight_ * std::min(rw, width_cap_) - deviation_weight_ * ro -
      heading_weight_ * static_cast<float>(facing) - bend_weight_ * static_cast<float>(bend);
    if (lok && !segmentFree(rx, ry, lwx, lwy)) {
      ls -= reach_penalty_;
    }
    if (rok && !segmentFree(rx, ry, rwx, rwy)) {
      rs -= reach_penalty_;
    }
    if (!have_scores_) {
      left_score_ = ls;
      right_score_ = rs;
      have_scores_ = true;
    } else {
      if (lok) {
        left_score_ += score_alpha_ * (ls - left_score_);
      }
      if (rok) {
        right_score_ += score_alpha_ * (rs - right_score_);
      }
    }

    float chosen = side_;
    const char * reason = "";
    if (side_ == 0.0f) {
      const bool remembered = memory_side_ != 0.0f && now - memory_time_ < side_memory_time_ &&
        std::hypot(bx - memory_x_, by - memory_y_) < side_memory_distance_ &&
        (memory_side_ > 0.0f ? lok : rok);
      if (fresh && remembered) {
        chosen = memory_side_;
        reason = "remembered";
      } else if (lok || rok) {
        chosen = (lok && (!rok || left_score_ >= right_score_)) ? 1.0f : -1.0f;
        reason = "best score";
      }
    } else {
      const bool mine_ok = side_ > 0.0f ? lok : rok;
      const bool other_ok = side_ > 0.0f ? rok : lok;
      if (mine_ok) {
        side_ok_time_ = now;
      }
      const float mine = side_ > 0.0f ? left_score_ : right_score_;
      const float other = side_ > 0.0f ? right_score_ : left_score_;
      if (other_ok && now - side_ok_time_ > side_lost_time_) {
        chosen = -side_;
        reason = "gap lost";
      } else if (other_ok && now - side_time_ > side_hold_time_ && other > mine + switch_margin_) {
        chosen = -side_;
        reason = "clearly better";
      }
    }
    if (chosen != side_) {
      RCLCPP_INFO(
        logger_,
        "\033[38;5;208mDetourCritic side %s (%s) | obstacle %.2f m, path bend %+.2f | left offset %.2f width %.2f "
        "score %.2f | right offset %.2f width %.2f score %.2f\033[0m",
        chosen > 0.0f ? "left" : (chosen < 0.0f ? "right" : "none"), reason, length, bend, lo, lw,
        left_score_, ro, rw, right_score_);
      side_ = chosen;
      side_time_ = now;
      side_ok_time_ = now;
      have_wp_ = false;
    }
    if (side_ != 0.0f) {
      memory_x_ = bx;
      memory_y_ = by;
      const bool ok = side_ > 0.0f ? lok : rok;
      if (ok) {
        const double wx = side_ > 0.0f ? lwx : rwx;
        const double wy = side_ > 0.0f ? lwy : rwy;
        if (!have_wp_ || std::hypot(wx - wp_x_, wy - wp_y_) > refresh_distance_) {
          wp_x_ = wx;
          wp_y_ = wy;
          have_wp_ = true;
        }
        const double lat = side_ > 0.0f ? l_lat : -r_lat;
        const double robot_lat = (rx - bx) * nx + (ry - by) * ny;
        exit_x_ = rx + (lat - robot_lat) * nx;
        exit_y_ = ry + (lat - robot_lat) * ny;
        have_exit_ = true;
      }
    }
  }
  if (!have_wp_) {
    return;
  }

  const float k = level(now);
  if (k <= 0.0f) {
    return;
  }
  bool direct = true;
  float gx = 0.0f, gy = 0.0f;
  float gain = 1.0f;
  float align = align_deg_;
  if (use_routes_) {
    if (wps_.empty()) {
      return;
    }
    gain = pre_wp1_gain_;
    const double rst = (rx - block_x_) * block_tx_ + (ry - block_y_) * block_ty_;
    const double w1x = straight_segments_ ? wp1_x_ : wps_[0][0];
    const double w1y = straight_segments_ ? wp1_y_ : wps_[0][1];
    const double w1s = straight_segments_ ? wp1_st_ : wps_[0][2];
    const bool reached = rst >= w1s - reach_ahead_ ||
      std::hypot(rx - w1x, ry - w1y) < reach_radius_;
    if (reached && !passive_) {
      RCLCPP_INFO(logger_, "\033[38;5;208mDetourCritic first waypoint reached: passive\033[0m");
      passive_ = true;
    }
    size_t idx = wps_.size() - 1;
    for (size_t k = 0; k < wps_.size(); ++k) {
      if ((straight_segments_ || wps_[k][2] > rst + reach_ahead_) &&
        std::hypot(rx - wps_[k][0], ry - wps_[k][1]) > reach_radius_)
      {
        idx = k;
        break;
      }
    }
    double tgx = wps_[idx][0], tgy = wps_[idx][1];
    direct = segmentFree(rx, ry, tgx, tgy);
    if (!direct && route_used_.size() > 1) {
      size_t jr = 0, jt = 0;
      double br = 1e18, bt = 1e18;
      for (size_t j = 0; j < route_used_.size(); ++j) {
        const double dr = std::hypot(route_used_[j].first - rx, route_used_[j].second - ry);
        const double dt = std::hypot(route_used_[j].first - tgx, route_used_[j].second - tgy);
        if (dr < br) {
          br = dr;
          jr = j;
        }
        if (dt < bt) {
          bt = dt;
          jt = j;
        }
      }
      size_t pick = std::min(jr + 4, route_used_.size() - 1);
      for (size_t j = jt; j > jr; --j) {
        if (segmentFree(rx, ry, route_used_[j].first, route_used_[j].second)) {
          pick = j;
          break;
        }
      }
      tgx = route_used_[pick].first;
      tgy = route_used_[pick].second;
    }
    if (idx != last_target_ || direct != last_direct_) {
      RCLCPP_INFO(
        logger_, "\033[38;5;208mDetourCritic waypoint %zu/%zu (%s)%s\033[0m", idx + 1, wps_.size(),
        passive_ ? "passive" : "active", direct ? "" : " via route point, line blocked");
      last_target_ = idx;
      last_direct_ = direct;
    }
    gx = static_cast<float>(tgx);
    gy = static_cast<float>(tgy);
    if (debug_) {
      publishDebug(now, rx, ry, true, idx, direct, tgx, tgy, k);
    }
    if (passive_) {
      gain = passive_gain_;
      align = passive_align_deg_;
      direct = true;
    }
  } else {
    direct = !have_exit_ || segmentFree(rx, ry, wp_x_, wp_y_);
    gx = static_cast<float>(direct ? wp_x_ : exit_x_);
    gy = static_cast<float>(direct ? wp_y_ : exit_y_);
    if (direct != last_direct_) {
      RCLCPP_INFO(
        logger_, "\033[38;5;208mDetourCritic target %s\033[0m", direct ? "waypoint" : "side step (line to waypoint blocked)");
      last_direct_ = direct;
    }
  }
  const double bearing = std::atan2(gy - ry, gx - rx);
  const double err = std::atan2(std::sin(bearing - yaw), std::cos(bearing - yaw));
  const float dir = err > 0.0 ? 1.0f : -1.0f;
  const float exit_deg = (!passive_ && align_exit_deg_ >= 0.0f) ? std::min(align_exit_deg_, align) : align;
  const double err_deg = std::fabs(err) * 180.0 / M_PI;
  if (exit_deg >= align) {
    turning_ = err_deg > align;
  } else if (!turning_ && err_deg > align) {
    turning_ = true;
  } else if (turning_ && err_deg < exit_deg) {
    turning_ = false;
  }
  const bool turn_first = !direct || turning_;
  const size_t window = std::min(
    steps, std::max<size_t>(1, static_cast<size_t>(std::lround(horizon_s_ / data.model_dt))));
  const float tol = (turning_ ? exit_deg : align) * static_cast<float>(M_PI) / 180.0f;
  for (size_t i = 0; i < batch; ++i) {
    float dmin = 1e9f;
    for (size_t t = 0; t < steps; ++t) {
      const float d = std::hypot(traj_x(i, t) - gx, traj_y(i, t) - gy);
      dmin = std::min(dmin, d);
    }
    float c = attraction_weight_ * dmin;
    if (turn_first) {
      float r = 0.0f;
      for (size_t t = 0; t < window; ++t) {
        const float bt = std::atan2(gy - traj_y(i, t), gx - traj_x(i, t));
        float e = bt - traj_yaw(i, t);
        e = std::atan2(std::sin(e), std::cos(e));
        if (std::fabs(e) <= tol) {
          break;
        }
        r += forward_weight_ * std::max(0.0f, cvx(i, t)) +
          rotation_weight_ * std::max(0.0f, min_rotation_speed_ - dir * cwz(i, t));
      }
      c += r * data.model_dt;
    }
    data.costs(i) += k * gain * c;
  }
}

void DetourCritic::deactivate(const char * why, double now, double, double)
{
  if (active_) {
    RCLCPP_INFO(logger_, "\033[38;5;208mDetourCritic off (%s)\033[0m", why);
    if (side_ != 0.0f) {
      memory_side_ = side_;
      memory_time_ = now;
    }
  }
  active_ = false;
  side_ = 0.0f;
  have_wp_ = false;
  have_exit_ = false;
  have_scores_ = false;
  have_block_ = false;
  passive_ = false;
  turning_ = false;
  wps_.clear();
  route_used_.clear();
}

int DetourCritic::cost(double x, double y) const
{
  unsigned int mx, my;
  if (!costmap_->worldToMap(x, y, mx, my)) {
    return nav2_costmap_2d::NO_INFORMATION;
  }
  return costmap_->getCost(mx, my);
}

bool DetourCritic::nearObstacle(double rx, double ry, double yaw) const
{
  const double c = std::cos(yaw), s = std::sin(yaw);
  const double hl = half_length_ + near_clearance_;
  const double hw = half_width_ + 0.10;
  const double step = costmap_->getResolution();
  for (double u = 0.0; u <= hl + 1e-6; u += step) {
    for (double w = -hw; w <= hw + 1e-6; w += step) {
      const int k = cost(rx + u * c - w * s, ry + u * s + w * c);
      if (k >= nav2_costmap_2d::LETHAL_OBSTACLE && k != nav2_costmap_2d::NO_INFORMATION) {
        return true;
      }
    }
  }
  return false;
}

bool DetourCritic::nearMover(double x, double y, const std::vector<Mover> & movers) const
{
  for (const auto & m : movers) {
    if (std::hypot(x - m.x, y - m.y) < mover_radius_) {
      return true;
    }
  }
  return false;
}

bool DetourCritic::segmentFree(double x0, double y0, double x1, double y1) const
{
  const double len = std::hypot(x1 - x0, y1 - y0);
  const double step = costmap_->getResolution();
  const int n = std::max(1, static_cast<int>(std::ceil(len / step)));
  for (int k = 1; k <= n; ++k) {
    const double f = static_cast<double>(k) / n;
    const int c = cost(x0 + f * (x1 - x0), y0 + f * (y1 - y0));
    if (c >= nav2_costmap_2d::INSCRIBED_INFLATED_OBSTACLE && c != nav2_costmap_2d::NO_INFORMATION) {
      return false;
    }
  }
  return true;
}

void DetourCritic::findBlocked(
  CriticData & data, double rx, double ry, const std::vector<Mover> & movers,
  size_t & blocked, double & tx, double & ty) const
{
  const auto & path = data.path;
  const size_t n = path.x.shape(0);
  size_t closest = 0;
  double best = 1e9;
  for (size_t i = 0; i < n; ++i) {
    const double d = std::hypot(path.x(i) - rx, path.y(i) - ry);
    if (d < best) {
      best = d;
      closest = i;
    }
  }
  double s = 0.0;
  for (size_t i = closest + 1; i < n; ++i) {
    s += std::hypot(path.x(i) - path.x(i - 1), path.y(i) - path.y(i - 1));
    if (s > check_distance_) {
      return;
    }
    if (s < min_distance_) {
      continue;
    }
    const size_t a = i - 1;
    const size_t b = std::min(i + 1, n - 1);
    double ux = path.x(b) - path.x(a);
    double uy = path.y(b) - path.y(a);
    const double un = std::hypot(ux, uy);
    if (un < 1e-6) {
      continue;
    }
    ux /= un;
    uy /= un;
    for (double off = -corridor_half_width_; off <= corridor_half_width_ + 1e-6; off += 0.10) {
      const double cx = path.x(i) - uy * off;
      const double cy = path.y(i) + ux * off;
      const int c = cost(cx, cy);
      if (c >= block_cost_ && c != nav2_costmap_2d::NO_INFORMATION && !nearMover(cx, cy, movers)) {
        blocked = i;
        tx = ux;
        ty = uy;
        return;
      }
    }
  }
}

double DetourCritic::obstacleLength(
  double bx, double by, double tx, double ty, double nx, double ny,
  const std::vector<Mover> & movers) const
{
  const double step = costmap_->getResolution();
  double last = 0.0;
  for (double s = 0.0; s <= obstacle_max_length_ + 1e-6; s += step) {
    bool hit = false;
    for (double off = -corridor_half_width_; off <= corridor_half_width_ + 1e-6; off += 0.10) {
      const double cx = bx + s * tx + off * nx;
      const double cy = by + s * ty + off * ny;
      const int c = cost(cx, cy);
      if (c >= block_cost_ && c != nav2_costmap_2d::NO_INFORMATION && !nearMover(cx, cy, movers)) {
        hit = true;
        break;
      }
    }
    if (hit) {
      last = s;
    } else if (s - last > free_length_) {
      break;
    }
  }
  return last;
}

bool DetourCritic::sideGap(
  double bx, double by, double tx, double ty, double nx, double ny, double side, double length,
  float & offset, float & width) const
{
  offset = 0.0f;
  width = 1e9f;
  const int stations = std::max(1, static_cast<int>(std::ceil(length / station_step_)) + 1);
  for (int k = 0; k < stations; ++k) {
    const double s = std::min(length, k * static_cast<double>(station_step_));
    float o, w;
    findGap(bx + s * tx, by + s * ty, tx, ty, nx, ny, side, o, w);
    if (o < 0.0f) {
      offset = -1.0f;
      width = 0.0f;
      return false;
    }
    offset = std::max(offset, o);
    width = std::min(width, w);
  }
  return true;
}

double DetourCritic::pathBend(CriticData & data, size_t blocked, double length, double nx, double ny) const
{
  const auto & path = data.path;
  const size_t n = path.x.shape(0);
  const double target = length + bend_lookahead_;
  double s = 0.0;
  size_t i = blocked;
  while (i + 1 < n && s < target) {
    s += std::hypot(path.x(i + 1) - path.x(i), path.y(i + 1) - path.y(i));
    ++i;
  }
  if (s < 1e-3) {
    return 0.0;
  }
  const double lat = (path.x(i) - path.x(blocked)) * nx + (path.y(i) - path.y(blocked)) * ny;
  return std::clamp(lat / s, -1.0, 1.0);
}

void DetourCritic::findGap(
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


bool DetourCritic::route(
  double rx, double ry, double gx, double gy, double bx, double by, double tx, double ty,
  double nx, double ny, double length, double side, std::vector<std::pair<double, double>> & out,
  double & total, double & mean_cost) const
{
  out.clear();
  total = std::numeric_limits<double>::infinity();
  mean_cost = 0.0;
  const unsigned int W = costmap_->getSizeInCellsX();
  const unsigned int H = costmap_->getSizeInCellsY();
  const double res = costmap_->getResolution();
  unsigned int sx, sy, ex, ey;
  if (!costmap_->worldToMap(rx, ry, sx, sy) || !costmap_->worldToMap(gx, gy, ex, ey)) {
    return false;
  }
  const size_t N = static_cast<size_t>(W) * H;
  std::vector<double> dist(N, std::numeric_limits<double>::infinity());
  std::vector<int> prev(N, -1);
  using Item = std::pair<double, size_t>;
  std::priority_queue<Item, std::vector<Item>, std::greater<Item>> pq;
  const size_t s0 = static_cast<size_t>(sy) * W + sx;
  const size_t g0 = static_cast<size_t>(ey) * W + ex;
  dist[s0] = 0.0;
  pq.push({0.0, s0});
  const int dx[8] = {1, -1, 0, 0, 1, 1, -1, -1};
  const int dy[8] = {0, 0, 1, -1, 1, -1, 1, -1};
  while (!pq.empty()) {
    const auto [d, k] = pq.top();
    pq.pop();
    if (d > dist[k]) {
      continue;
    }
    if (k == g0) {
      break;
    }
    const int x = static_cast<int>(k % W);
    const int y = static_cast<int>(k / W);
    for (int n = 0; n < 8; ++n) {
      const int qx = x + dx[n];
      const int qy = y + dy[n];
      if (qx < 0 || qy < 0 || qx >= static_cast<int>(W) || qy >= static_cast<int>(H)) {
        continue;
      }
      double wx, wy;
      costmap_->mapToWorld(qx, qy, wx, wy);
      const double near_robot = std::hypot(wx - rx, wy - ry);
      unsigned char c = costmap_->getCost(qx, qy);
      if (c == nav2_costmap_2d::NO_INFORMATION) {
        c = 128;
      }
      const bool escape = c == nav2_costmap_2d::INSCRIBED_INFLATED_OBSTACLE && near_robot < half_length_;
      if (c >= nav2_costmap_2d::INSCRIBED_INFLATED_OBSTACLE && !escape) {
        continue;
      }
      const double st = (wx - bx) * tx + (wy - by) * ty;
      const double la = (wx - bx) * nx + (wy - by) * ny;
      if (near_robot > half_length_ && st > -0.3 && st < length + 0.3 && la * side < 0.0) {
        continue;
      }
      const double step = (n < 4 ? 1.0 : M_SQRT2) * res;
      const double nd = d + step * (1.0 + route_cost_weight_ * std::min<int>(c, 252) / 252.0);
      const size_t q = static_cast<size_t>(qy) * W + qx;
      if (nd < dist[q]) {
        dist[q] = nd;
        prev[q] = static_cast<int>(k);
        pq.push({nd, q});
      }
    }
  }
  if (!std::isfinite(dist[g0])) {
    return false;
  }
  total = dist[g0];
  double sum = 0.0;
  size_t cnt = 0;
  for (int k = static_cast<int>(g0); k >= 0; k = prev[k]) {
    double wx, wy;
    costmap_->mapToWorld(k % W, k / W, wx, wy);
    out.push_back({wx, wy});
    unsigned char c = costmap_->getCost(k % W, k / W);
    sum += (c == nav2_costmap_2d::NO_INFORMATION) ? 128.0 : static_cast<double>(c);
    ++cnt;
    if (static_cast<size_t>(k) == s0) {
      break;
    }
  }
  std::reverse(out.begin(), out.end());
  mean_cost = cnt ? sum / cnt : 0.0;
  return true;
}

bool DetourCritic::goalBeyond(
  CriticData & data, size_t blocked, double length, double & gx, double & gy) const
{
  const auto & path = data.path;
  const size_t n = path.x.shape(0);
  bool have = false;
  double s = 0.0;
  for (size_t i = blocked + 1; i < n; ++i) {
    s += std::hypot(path.x(i) - path.x(i - 1), path.y(i) - path.y(i - 1));
    unsigned int mx, my;
    if (!costmap_->worldToMap(path.x(i), path.y(i), mx, my)) {
      break;
    }
    const int c = costmap_->getCost(mx, my);
    if (c == nav2_costmap_2d::NO_INFORMATION || c >= nav2_costmap_2d::INSCRIBED_INFLATED_OBSTACLE) {
      continue;
    }
    if (s > length) {
      gx = path.x(i);
      gy = path.y(i);
      have = true;
      if (s > length + route_goal_margin_) {
        break;
      }
    }
  }
  return have;
}

void DetourCritic::routeSides(
  CriticData & data, double rx, double ry, size_t blocked, double bx, double by, double tx,
  double ty, double nx, double ny, double length, double now)
{
  const bool new_block = !have_block_ || std::hypot(bx - block_x_, by - block_y_) > route_refresh_distance_;
  const bool fresh_route = new_block || side_ == 0.0f || now - route_time_ >= route_period_ ||
    std::hypot(rx - route_x_, ry - route_y_) > route_refresh_distance_;
  if (!fresh_route) {
    return;
  }
  route_time_ = now;
  route_x_ = rx;
  route_y_ = ry;
  double gx = 0.0, gy = 0.0;
  if (!goalBeyond(data, blocked, length, gx, gy)) {
    route_okl_ = false;
    route_okr_ = false;
    route_l_.clear();
    route_r_.clear();
    return;
  }
  route_okl_ = route(rx, ry, gx, gy, bx, by, tx, ty, nx, ny, length, 1.0, route_l_, route_cl_, route_ml_);
  route_okr_ = route(rx, ry, gx, gy, bx, by, tx, ty, nx, ny, length, -1.0, route_r_, route_cr_, route_mr_);
}

bool DetourCritic::corridorFree(
  double x0, double y0, double x1, double y1, double half_width, double skip,
  int threshold) const
{
  const double len = std::hypot(x1 - x0, y1 - y0);
  if (len < 1e-6) {
    return true;
  }
  const double tx = (x1 - x0) / len, ty = (y1 - y0) / len;
  const double step = costmap_->getResolution();
  const int n = std::max(1, static_cast<int>(std::ceil(len / step)));
  const int m = static_cast<int>(std::ceil(half_width / step));
  for (int k = 0; k <= n; ++k) {
    const double f = len * k / n;
    if (f < skip) {
      continue;
    }
    for (int w = -m; w <= m; ++w) {
      const double o = std::max(-half_width, std::min(half_width, w * step));
      const int c = cost(x0 + f * tx - o * ty, y0 + f * ty + o * tx);
      if (c >= threshold && c != nav2_costmap_2d::NO_INFORMATION) {
        return false;
      }
    }
  }
  return true;
}

bool DetourCritic::turnFree(double x, double y) const
{
  const double rad = std::hypot(half_length_, half_width_) + turn_margin_;
  const double step = costmap_->getResolution();
  const int m = static_cast<int>(std::ceil(rad / step));
  for (int i = -m; i <= m; ++i) {
    for (int j = -m; j <= m; ++j) {
      if (std::hypot(i * step, j * step) > rad) {
        continue;
      }
      if (cost(x + i * step, y + j * step) == nav2_costmap_2d::LETHAL_OBSTACLE) {
        return false;
      }
    }
  }
  return true;
}

void DetourCritic::buildWaypoints(const std::vector<std::pair<double, double>> & r)
{
  wps_.clear();
  route_used_ = r;
  if (r.size() < 2) {
    return;
  }
  const auto station = [this](const std::pair<double, double> & p) {
      return (p.first - block_x_) * block_tx_ + (p.second - block_y_) * block_ty_;
    };
  size_t j1 = r.size() - 1;
  for (size_t j = 0; j < r.size(); ++j) {
    if (station(r[j]) >= 0.0) {
      j1 = j;
      break;
    }
  }
  if (straight_segments_) {
    wp1_x_ = r[j1].first;
    wp1_y_ = r[j1].second;
    wp1_st_ = station(r[j1]);
    const double hw = half_width_ + segment_margin_;
    size_t a = 0;
    while (a + 1 < r.size()) {
      const double skip = a == 0 ? half_length_ : 0.0;
      size_t best = a + 1;
      for (size_t j = a + 2; j < r.size(); ++j) {
        if (!corridorFree(
            r[a].first, r[a].second, r[j].first, r[j].second, hw, skip,
            nav2_costmap_2d::LETHAL_OBSTACLE))
        {
          break;
        }
        best = j;
      }
      if (best == a + 1) {
        for (size_t j = a + 2; j < r.size(); ++j) {
          if (!corridorFree(
              r[a].first, r[a].second, r[j].first, r[j].second, 0.0, skip,
              nav2_costmap_2d::INSCRIBED_INFLATED_OBSTACLE))
          {
            break;
          }
          best = j;
        }
      }
      if (best + 1 < r.size()) {
        for (size_t j = best; j > a + 1; --j) {
          if (turnFree(r[j].first, r[j].second)) {
            best = j;
            break;
          }
        }
      }
      wps_.push_back({r[best].first, r[best].second, station(r[best])});
      a = best;
    }
    return;
  }
  wps_.push_back({r[j1].first, r[j1].second, station(r[j1])});
  double acc = 0.0;
  for (size_t j = j1 + 1; j < r.size(); ++j) {
    acc += std::hypot(r[j].first - r[j - 1].first, r[j].second - r[j - 1].second);
    if (acc >= waypoint_spacing_ || j + 1 == r.size()) {
      wps_.push_back({r[j].first, r[j].second, station(r[j])});
      acc = 0.0;
    }
  }
}

bool DetourCritic::blockFree() const
{
  if (!have_block_) {
    return true;
  }
  const double nx = -block_ty_;
  const double ny = block_tx_;
  const double step = costmap_->getResolution();
  for (double s = 0.0; s <= block_len_ + 1e-6; s += step) {
    for (double off = -corridor_half_width_; off <= corridor_half_width_ + 1e-6; off += 0.10) {
      const int c = cost(block_x_ + s * block_tx_ + off * nx, block_y_ + s * block_ty_ + off * ny);
      if (c >= block_cost_ && c != nav2_costmap_2d::NO_INFORMATION) {
        return false;
      }
    }
  }
  return true;
}

bool DetourCritic::routeBlockedAhead(double rx, double ry) const
{
  if (route_used_.size() < 2) {
    return false;
  }
  size_t jr = 0;
  double best = 1e18;
  for (size_t j = 0; j < route_used_.size(); ++j) {
    const double d = std::hypot(route_used_[j].first - rx, route_used_[j].second - ry);
    if (d < best) {
      best = d;
      jr = j;
    }
  }
  for (size_t j = jr + 1; j < route_used_.size(); ++j) {
    if (std::hypot(route_used_[j].first - rx, route_used_[j].second - ry) < half_length_) {
      continue;
    }
    const int c = cost(route_used_[j].first, route_used_[j].second);
    if (c >= nav2_costmap_2d::INSCRIBED_INFLATED_OBSTACLE && c != nav2_costmap_2d::NO_INFORMATION) {
      return true;
    }
  }
  return false;
}

void DetourCritic::publishDebug(
  double now, double rx, double ry, bool target, size_t idx, bool direct, double tgx, double tgy,
  float k)
{
  if (!debug_pub_ || now - debug_time_ < 0.2) {
    return;
  }
  debug_time_ = now;
  visualization_msgs::msg::MarkerArray arr;
  const auto stamp = parent_.lock()->now();
  const auto line = [&](const std::string & ns, const std::vector<std::pair<double, double>> & r,
      float red, float green, float blue) {
      visualization_msgs::msg::Marker m;
      m.header.frame_id = frame_;
      m.header.stamp = stamp;
      m.ns = ns;
      m.id = 0;
      m.type = visualization_msgs::msg::Marker::LINE_STRIP;
      m.action = visualization_msgs::msg::Marker::ADD;
      m.scale.x = 0.04;
      m.color.r = red;
      m.color.g = green;
      m.color.b = blue;
      m.color.a = 1.0;
      m.pose.orientation.w = 1.0;
      for (const auto & p : r) {
        geometry_msgs::msg::Point q;
        q.x = p.first;
        q.y = p.second;
        m.points.push_back(q);
      }
      arr.markers.push_back(m);
    };
  line("route_left", route_okl_ ? route_l_ : std::vector<std::pair<double, double>>{}, 0.1f, 0.4f, 1.0f);
  line("route_right", route_okr_ ? route_r_ : std::vector<std::pair<double, double>>{}, 1.0f, 0.5f, 0.0f);
  std::vector<std::pair<double, double>> w;
  for (const auto & p : wps_) {
    w.push_back({p[0], p[1]});
  }
  line("waypoints", w, 0.0f, 0.7f, 0.0f);
  visualization_msgs::msg::Marker s;
  s.header.frame_id = frame_;
  s.header.stamp = stamp;
  s.ns = "state";
  s.id = 0;
  s.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
  s.action = visualization_msgs::msg::Marker::ADD;
  s.pose.position.x = rx;
  s.pose.position.y = ry + 1.0;
  s.pose.orientation.w = 1.0;
  s.scale.z = 0.2;
  s.color.a = 1.0;
  s.color.r = s.color.g = s.color.b = 1.0f;
  char buf[512];
  std::snprintf(
    buf, sizeof(buf),
    "{\"side\": %.0f, \"passive\": %d, \"target\": %d, \"idx\": %zu, \"n\": %zu, \"direct\": %d, "
    "\"tx\": %.3f, \"ty\": %.3f, \"level\": %.2f, \"bx\": %.3f, \"by\": %.3f, \"btx\": %.3f, "
    "\"bty\": %.3f, \"len\": %.2f, \"okl\": %d, \"okr\": %d, \"cl\": %.1f, \"cr\": %.1f, "
    "\"ml\": %.0f, \"mr\": %.0f}",
    side_, passive_ ? 1 : 0, target ? 1 : 0, idx, wps_.size(), direct ? 1 : 0, tgx, tgy, k,
    block_x_, block_y_, block_tx_, block_ty_, block_len_, route_okl_ ? 1 : 0, route_okr_ ? 1 : 0,
    route_okl_ ? route_cl_ : -1.0, route_okr_ ? route_cr_ : -1.0, route_ml_, route_mr_);
  s.text = buf;
  arr.markers.push_back(s);
  debug_pub_->publish(arr);
}

bool DetourCritic::passedBlock(double rx, double ry) const
{
  if (!use_routes_ || !have_block_) {
    return true;
  }
  const double rst = (rx - block_x_) * block_tx_ + (ry - block_y_) * block_ty_;
  return rst > block_len_ + pass_margin_ || blockFree() ||
         std::hypot(rx - block_x_, ry - block_y_) > check_distance_ + 1.5;
}
}

#include <pluginlib/class_list_macros.hpp>
PLUGINLIB_EXPORT_CLASS(mppi::critics::DetourCritic, mppi::critics::CriticFunction)
