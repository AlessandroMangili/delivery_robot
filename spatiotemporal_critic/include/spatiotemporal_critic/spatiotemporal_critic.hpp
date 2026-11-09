#ifndef SPATIOTEMPORAL_CRITIC__SPATIOTEMPORAL_CRITIC_HPP_
#define SPATIOTEMPORAL_CRITIC__SPATIOTEMPORAL_CRITIC_HPP_

#include <mutex>
#include <string>
#include <vector>

#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>

#include "nav2_mppi_controller/critic_function.hpp"
#include "dynamic_tracker_msgs/msg/track_array.hpp"
#include "builtin_interfaces/msg/time.hpp"
#include "visualization_msgs/msg/marker_array.hpp"

namespace mppi::critics
{

struct TrackSnapshot
{
  double x, y;
  double vx, vy;
  double pos_std;
  double vel_std;
};

class SpatioTemporalCritic : public mppi::critics::CriticFunction
{
public:
  void initialize() override;

  void score(mppi::CriticData & data) override;

protected:
  void tracksCallback(const dynamic_tracker_msgs::msg::TrackArray::SharedPtr msg);

  struct RobotRef
  {
    double x, y, cyaw, syaw, v;
  };

  bool isConsidered(double px, double py, double t, const RobotRef & r) const;

  void publishPredictions(
    const std::vector<TrackSnapshot> & tracks,
    float dt, size_t time, const RobotRef & r);

  void publishRobotTrajectory(
    const mppi::CriticData & data, float dt, size_t time);

  bool enabled_{true};
  std::string tracks_topic_;

  unsigned int power_{1};
  float weight_{20.0f};
  float collision_radius_{0.5f};
  float vel_std_gain_{0.4f};
  float min_ped_speed_{0.4f};
  float max_vel_std_{0.8f};
  float max_track_age_s_{0.5f};

  bool rear_filter_{false};
  bool rear_use_prediction_{false};
  float rear_margin_m_{0.3f};
  bool rear_keep_close_{true};
  float rear_keep_radius_m_{1.0f};
  bool rear_on_reverse_{true};

  float prediction_horizon_s_{3.0f};
  float max_radius_{1.5f};

  bool worst_case_{true};

  std::string cost_mode_;
  bool use_overlap_{true};
  bool use_ttc_{true};

  float ttc_weight_{20.0f};
  float ttc_power_{1.0f};
  float ttc_epsilon_{0.1f};

  bool publish_predictions_{true};
  std::string predictions_topic_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pred_pub_;

  bool publish_robot_traj_{true};
  std::string robot_traj_topic_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr robot_pub_;

  std::string world_frame_;

  std::mutex tracks_mutex_;
  std::vector<TrackSnapshot> tracks_;
  builtin_interfaces::msg::Time tracks_stamp_;
  std::string tracks_frame_{"odom"};

  rclcpp::Subscription<dynamic_tracker_msgs::msg::TrackArray>::SharedPtr tracks_sub_;

  unsigned int diag_period_calls_{0};
  unsigned int score_calls_{0};
};

}

#endif
