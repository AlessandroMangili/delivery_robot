#include "mppi_critics/rotation_deadzone_critic.hpp"

#include <algorithm>
#include <cmath>

namespace mppi::critics
{

void RotationDeadzoneCritic::initialize()
{
  auto getParam = parameters_handler_->getParamGetter(name_);
  getParam(power_, "cost_power", 1);
  getParam(weight_, "cost_weight", 10.0f);
  getParam(v_max_, "v_max", 0.08f);
  getParam(w_min_, "w_min", 0.06f);
  getParam(stall_w_, "stall_w", 0.19f);
  getParam(window_s_, "window_s", 1.0f);
  getParam(brake_v_, "brake_v", 0.0f);

  RCLCPP_INFO(
    logger_,
    "RotationDeadzoneCritic: weight %.1f | when MPPI rotates in place (|v| < %.2f m/s, |w| > %.2f rad/s) "
    "it pushes |w| >= %.2f rad/s in the first %.1f s, brake v %.1f",
    weight_, v_max_, w_min_, stall_w_, window_s_, brake_v_);
}

void RotationDeadzoneCritic::score(mppi::CriticData & data)
{
  if (!enabled_) {
    return;
  }

  const auto & cvx = data.state.cvx;
  const auto & cwz = data.state.cwz;
  const size_t batch = cvx.shape(0);
  const size_t time = cvx.shape(1);
  if (batch == 0 || time == 0 || cwz.shape(0) != batch || cwz.shape(1) != time ||
    data.costs.shape(0) != batch || data.model_dt <= 0.0f)
  {
    return;
  }

  const size_t steps = std::min(
    time, std::max<size_t>(1, static_cast<size_t>(std::lround(window_s_ / data.model_dt))));
  const float inv_steps = 1.0f / static_cast<float>(steps);

  double sum_v = 0.0, sum_w = 0.0;
  for (size_t i = 0; i < batch; ++i) {
    for (size_t t = 0; t < steps; ++t) {
      sum_v += cvx(i, t);
      sum_w += cwz(i, t);
    }
  }
  const double n = static_cast<double>(batch * steps);
  const float mean_v = static_cast<float>(sum_v / n);
  const float mean_w = static_cast<float>(sum_w / n);
  if (std::fabs(mean_w) < w_min_ || std::fabs(mean_v) >= v_max_) {
    return;
  }

  const float dir = mean_w > 0.0f ? 1.0f : -1.0f;
  for (size_t i = 0; i < batch; ++i) {
    float w = 0.0f;
    float v = 0.0f;
    for (size_t t = 0; t < steps; ++t) {
      w += cwz(i, t);
      v += std::fabs(cvx(i, t));
    }
    w = dir * w * inv_steps;
    v *= inv_steps;
    const float penalty = std::max(0.0f, stall_w_ - w);
    float c = weight_ * penalty + brake_v_ * v;
    if (c <= 0.0f) {
      continue;
    }
    if (power_ > 1) {
      c = std::pow(c, static_cast<float>(power_));
    }
    data.costs(i) += c;
  }
}

}

#include <pluginlib/class_list_macros.hpp>
PLUGINLIB_EXPORT_CLASS(
  mppi::critics::RotationDeadzoneCritic,
  mppi::critics::CriticFunction)
