#include "mppi_critics/clearance_critic.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>

#include "nav2_costmap_2d/cost_values.hpp"

namespace mppi::critics
{

namespace
{

void edt1d(const std::vector<float> & f, std::vector<float> & d, int n)
{
  std::vector<int> v(n);
  std::vector<float> z(n + 1);
  int k = 0;
  v[0] = 0;
  z[0] = -std::numeric_limits<float>::infinity();
  z[1] = std::numeric_limits<float>::infinity();
  for (int q = 1; q < n; ++q) {
    float s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2.0f * q - 2.0f * v[k]);
    while (s <= z[k]) {
      --k;
      s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2.0f * q - 2.0f * v[k]);
    }
    ++k;
    v[k] = q;
    z[k] = s;
    z[k + 1] = std::numeric_limits<float>::infinity();
  }
  k = 0;
  for (int q = 0; q < n; ++q) {
    while (z[k + 1] < q) {
      ++k;
    }
    d[q] = (q - v[k]) * (q - v[k]) + f[v[k]];
  }
}

}

void ClearanceCritic::initialize()
{
  auto getParam = parameters_handler_->getParamGetter(name_);
  getParam(weight_, "cost_weight", 30.0f);
  getParam(safe_distance_, "safe_distance", 0.30f);
  getParam(perimeter_step_, "perimeter_step", 0.25f);
  getParam(trajectory_point_step_, "trajectory_point_step", 3);
  getParam(obstacle_cost_, "obstacle_cost", 254);
  RCLCPP_INFO(
    logger_, "ClearanceCritic: penalises footprint edge closer than %.2f m to cells >= %d, weight %.1f",
    safe_distance_, obstacle_cost_, weight_);
}

void ClearanceCritic::updateDistanceField()
{
  size_x_ = costmap_->getSizeInCellsX();
  size_y_ = costmap_->getSizeInCellsY();
  resolution_ = costmap_->getResolution();
  origin_x_ = costmap_->getOriginX();
  origin_y_ = costmap_->getOriginY();
  const float big = 1e10f;
  const int nx = static_cast<int>(size_x_);
  const int ny = static_cast<int>(size_y_);
  field_.assign(static_cast<size_t>(nx) * ny, big);
  const unsigned char * grid = costmap_->getCharMap();
  for (int i = 0; i < nx * ny; ++i) {
    const unsigned char c = grid[i];
    if (c >= obstacle_cost_ && c != nav2_costmap_2d::NO_INFORMATION) {
      field_[i] = 0.0f;
    }
  }
  const int n = std::max(nx, ny);
  std::vector<float> f(n), d(n);
  for (int x = 0; x < nx; ++x) {
    for (int y = 0; y < ny; ++y) {
      f[y] = field_[y * nx + x];
    }
    edt1d(f, d, ny);
    for (int y = 0; y < ny; ++y) {
      field_[y * nx + x] = d[y];
    }
  }
  for (int y = 0; y < ny; ++y) {
    for (int x = 0; x < nx; ++x) {
      f[x] = field_[y * nx + x];
    }
    edt1d(f, d, nx);
    for (int x = 0; x < nx; ++x) {
      field_[y * nx + x] = std::sqrt(d[x]) * static_cast<float>(resolution_);
    }
  }
}

float ClearanceCritic::distanceAt(float wx, float wy) const
{
  const int mx = static_cast<int>(std::floor((wx - origin_x_) / resolution_));
  const int my = static_cast<int>(std::floor((wy - origin_y_) / resolution_));
  if (mx < 0 || my < 0 || mx >= static_cast<int>(size_x_) || my >= static_cast<int>(size_y_)) {
    return safe_distance_;
  }
  return std::max(0.0f, field_[my * size_x_ + mx] - 0.5f * static_cast<float>(resolution_));
}

void ClearanceCritic::score(mppi::CriticData & data)
{
  if (!enabled_) {
    return;
  }
  const double now = std::chrono::duration<double>(
    std::chrono::steady_clock::now().time_since_epoch()).count();
  if (now - last_field_time_ > 0.05 || field_.empty()) {
    updateDistanceField();
    last_field_time_ = now;
  }
  perimeter_.clear();
  const auto footprint = costmap_ros_->getRobotFootprint();
  for (size_t k = 0; k < footprint.size(); ++k) {
    const auto & a = footprint[k];
    const auto & b = footprint[(k + 1) % footprint.size()];
    const double len = std::hypot(b.x - a.x, b.y - a.y);
    const int n = std::max(1, static_cast<int>(std::ceil(len / perimeter_step_)));
    for (int s = 0; s < n; ++s) {
      const double u = static_cast<double>(s) / n;
      perimeter_.emplace_back(a.x + u * (b.x - a.x), a.y + u * (b.y - a.y));
    }
  }
  if (perimeter_.empty() || field_.empty()) {
    return;
  }
  const auto & traj = data.trajectories;
  const size_t batch = traj.x.shape(0);
  const size_t steps = traj.x.shape(1);
  const size_t stride = static_cast<size_t>(std::max(1, trajectory_point_step_));
  const float inv_safe = 1.0f / std::max(0.01f, safe_distance_);
  for (size_t i = 0; i < batch; ++i) {
    float total = 0.0f;
    size_t count = 0;
    for (size_t t = 0; t < steps; t += stride) {
      const float x = traj.x(i, t);
      const float y = traj.y(i, t);
      const float c = std::cos(traj.yaws(i, t));
      const float s = std::sin(traj.yaws(i, t));
      float dmin = safe_distance_;
      for (const auto & p : perimeter_) {
        const float px = x + p.first * c - p.second * s;
        const float py = y + p.first * s + p.second * c;
        dmin = std::min(dmin, distanceAt(px, py));
      }
      const float pen = (safe_distance_ - dmin) * inv_safe;
      total += pen * pen;
      ++count;
    }
    if (count > 0 && total > 0.0f) {
      data.costs(i) += weight_ * total / static_cast<float>(count);
    }
  }
}

}

#include <pluginlib/class_list_macros.hpp>
PLUGINLIB_EXPORT_CLASS(mppi::critics::ClearanceCritic, mppi::critics::CriticFunction)
