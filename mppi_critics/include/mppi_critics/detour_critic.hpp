#ifndef MPPI_CRITICS__DETOUR_CRITIC_HPP_
#define MPPI_CRITICS__DETOUR_CRITIC_HPP_

#include <array>
#include <deque>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "nav2_mppi_controller/critic_function.hpp"
#include "dynamic_tracker_msgs/msg/track_array.hpp"
#include "visualization_msgs/msg/marker_array.hpp"

namespace mppi::critics
{

class DetourCritic : public CriticFunction
{
public:
  void initialize() override;

  void score(CriticData & data) override;

protected:
  struct Mover
  {
    double x;
    double y;
  };

  void deactivate(const char * why, double now, double rx, double ry);

  int cost(double x, double y) const;

  bool nearObstacle(double rx, double ry, double yaw) const;

  bool nearMover(double x, double y, const std::vector<Mover> & movers) const;

  void findBlocked(
    CriticData & data, double rx, double ry, const std::vector<Mover> & movers,
    size_t & blocked, double & tx, double & ty) const;

  void findGap(
    double bx, double by, double tx, double ty, double nx, double ny, double side,
    float & offset, float & width) const;

  double obstacleLength(
    double bx, double by, double tx, double ty, double nx, double ny,
    const std::vector<Mover> & movers) const;

  bool sideGap(
    double bx, double by, double tx, double ty, double nx, double ny, double side, double length,
    float & offset, float & width) const;

  double pathBend(CriticData & data, size_t blocked, double length, double nx, double ny) const;

  bool route(
    double rx, double ry, double gx, double gy, double bx, double by, double tx, double ty,
    double nx, double ny, double length, double side, std::vector<std::pair<double, double>> & out,
    double & total, double & mean_cost) const;

  bool goalBeyond(CriticData & data, size_t blocked, double length, double & gx, double & gy) const;

  void routeSides(
    CriticData & data, double rx, double ry, size_t blocked, double bx, double by, double tx,
    double ty, double nx, double ny, double length, double now);

  void buildWaypoints(const std::vector<std::pair<double, double>> & r);

  bool blockFree() const;

  bool routeBlockedAhead(double rx, double ry) const;

  bool passedBlock(double rx, double ry) const;

  float level(double now) const;

  bool segmentFree(double x0, double y0, double x1, double y1) const;

  bool corridorFree(

    double x0, double y0, double x1, double y1, double half_width, double skip,

    int threshold) const;

  bool turnFree(double x, double y) const;

  void tracksCallback(const dynamic_tracker_msgs::msg::TrackArray::SharedPtr msg);

  std::vector<Mover> currentMovers(double now) const;

  void publishDebug(
    double now, double rx, double ry, bool target, size_t idx, bool direct, double tgx, double tgy,
    float k);

  float attraction_weight_{5.0f};
  float rotation_weight_{10.0f};
  float forward_weight_{10.0f};
  float min_rotation_speed_{0.25f};
  float horizon_s_{3.0f};
  float align_deg_{20.0f};
  float check_distance_{2.0f};
  float min_distance_{0.3f};
  float corridor_half_width_{0.20f};
  int block_cost_{254};
  int free_cost_{128};
  float max_offset_{1.2f};
  float min_gap_{0.8f};
  float width_cap_{2.0f};
  float path_half_length_{0.4f};
  float space_weight_{1.0f};
  float deviation_weight_{1.0f};
  float switch_margin_{0.5f};
  float progress_distance_{0.15f};
  float stuck_turn_deg_{20.0f};
  float stuck_time_{1.2f};
  float near_clearance_{0.25f};
  float release_time_{1.5f};
  float refresh_distance_{0.5f};
  float goal_distance_{1.0f};
  float half_length_{0.45f};
  float half_width_{0.33f};
  float ramp_time_{1.0f};
  float ramp_out_time_{1.5f};
  float station_step_{0.2f};
  float obstacle_max_length_{2.0f};
  float free_length_{0.3f};
  float bend_weight_{0.5f};
  float bend_lookahead_{1.5f};
  float side_hold_time_{3.0f};
  float side_lost_time_{1.0f};
  float side_memory_time_{8.0f};
  float side_memory_distance_{1.5f};
  float score_alpha_{0.3f};
  float heading_weight_{0.5f};
  float reach_penalty_{1.0f};
  float lateral_margin_{0.15f};
  float waypoint_ahead_{0.3f};
  bool ignore_movers_{true};
  bool use_routes_{true};
  float route_cost_weight_{10.0f};
  float route_switch_ratio_{0.8f};
  float route_mean_cost_{128.0f};
  float route_period_{1.0f};
  float route_refresh_distance_{0.3f};
  float route_goal_margin_{0.6f};
  float waypoint_spacing_{0.5f};
  float reach_ahead_{0.1f};
  float reach_radius_{0.35f};
  float passive_gain_{0.5f};
  float pre_wp1_gain_{1.0f};
  bool straight_segments_{false};
  float segment_margin_{0.10f};
  float turn_margin_{0.05f};
  float align_exit_deg_{-1.0f};
  bool turning_{false};
  double wp1_x_{0.0};
  double wp1_y_{0.0};
  double wp1_st_{0.0};
  float passive_align_deg_{60.0f};
  float pass_margin_{0.2f};
  float mover_speed_{0.3f};
  float mover_radius_{0.7f};
  float max_track_age_s_{1.0f};
  std::string tracks_topic_{"/dynamic_tracks_state"};

  bool active_{false};
  bool releasing_{false};
  double release_start_{0.0};
  float release_from_{0.0f};
  float side_{0.0f};
  bool have_wp_{false};
  bool have_exit_{false};
  bool last_direct_{true};
  double exit_x_{0.0};
  double exit_y_{0.0};
  double wp_x_{0.0};
  double wp_y_{0.0};
  double last_blocked_{-1e9};
  double on_time_{0.0};
  double side_time_{-1e9};
  double side_ok_time_{-1e9};
  bool have_scores_{false};
  float left_score_{0.0f};
  float right_score_{0.0f};
  float memory_side_{0.0f};
  double memory_time_{-1e9};
  double memory_x_{0.0};
  double memory_y_{0.0};
  std::deque<std::array<double, 4>> poses_;
  double route_time_{-1e9};
  double route_x_{0.0};
  double route_y_{0.0};
  std::vector<std::pair<double, double>> route_l_;
  std::vector<std::pair<double, double>> route_r_;
  std::vector<std::pair<double, double>> route_used_;
  double route_cl_{0.0};
  double route_cr_{0.0};
  double route_ml_{0.0};
  double route_mr_{0.0};
  bool route_okl_{false};
  bool route_okr_{false};
  std::vector<std::array<double, 3>> wps_;
  bool have_block_{false};
  double block_x_{0.0};
  double block_y_{0.0};
  double block_tx_{1.0};
  double block_ty_{0.0};
  double block_len_{0.0};
  bool passive_{false};
  size_t last_target_{0};

  rclcpp::Subscription<dynamic_tracker_msgs::msg::TrackArray>::SharedPtr tracks_sub_;
  bool debug_{false};
  double debug_time_{-1e9};
  std::string frame_{"odom"};
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr debug_pub_;
  mutable std::mutex tracks_mutex_;
  std::vector<Mover> movers_;
  double movers_time_{-1e9};
};

}

#endif
