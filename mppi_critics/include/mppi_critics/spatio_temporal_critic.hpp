#ifndef MPPI_CRITICS__SPATIO_TEMPORAL_CRITIC_HPP_
#define MPPI_CRITICS__SPATIO_TEMPORAL_CRITIC_HPP_

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

  void publishPredictions(
    const std::vector<TrackSnapshot> & tracks,
    float dt, size_t time, float age_offset);

  std::string tracks_topic_;

  float weight_{20.0f};
  float collision_radius_{0.5f};
  float vel_std_gain_{0.4f};
  float min_ped_speed_{0.4f};
  float max_vel_std_{0.8f};
  float max_track_age_s_{0.5f};
  float marker_period_s_{0.2f};
  double last_marker_s_{-1.0};

  float prediction_horizon_s_{3.0f};
  float max_radius_{1.5f};
  float half_length_{0.45f};
  float half_width_{0.33f};

  float ttc_weight_{20.0f};
  float ttc_power_{1.0f};
  float ttc_epsilon_{0.1f};

  bool publish_predictions_{true};
  std::string predictions_topic_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pred_pub_;

  std::mutex tracks_mutex_;
  std::vector<TrackSnapshot> tracks_;
  builtin_interfaces::msg::Time tracks_stamp_;
  std::string tracks_frame_{"odom"};

  rclcpp::Subscription<dynamic_tracker_msgs::msg::TrackArray>::SharedPtr tracks_sub_;
};

}

#endif
