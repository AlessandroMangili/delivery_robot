#ifndef MPPI_CRITICS__HEADING_CRITIC_HPP_
#define MPPI_CRITICS__HEADING_CRITIC_HPP_

#include "nav2_mppi_controller/critic_function.hpp"

namespace mppi::critics
{

class HeadingCritic : public mppi::critics::CriticFunction
{
public:
  void initialize() override;
  void score(mppi::CriticData & data) override;

protected:
  int cost(double x, double y) const;
  void findGap(
    double bx, double by, double tx, double ty, double nx, double ny, double side,
    float & offset, float & width) const;

  float rotation_weight_{10.0f};
  float forward_weight_{20.0f};
  float min_rotation_speed_{0.20f};
  float horizon_s_{3.0f};
  float align_tolerance_deg_{15.0f};
  float activation_angle_deg_{30.0f};
  float deactivation_angle_deg_{10.0f};
  float heading_lookahead_{1.0f};
  float goal_distance_{0.5f};
  bool side_choice_{true};
  float check_distance_{2.0f};
  float min_distance_{0.3f};
  float corridor_half_width_{0.40f};
  int block_cost_{254};
  int free_cost_{128};
  float max_offset_{2.0f};
  float min_gap_{0.75f};
  float width_cap_{2.0f};
  float path_half_length_{0.4f};
  float space_weight_{1.0f};
  float deviation_weight_{1.0f};
  float switch_margin_{0.5f};
  float lost_side_time_{1.0f};
  float release_time_{2.0f};
  float side_{0.0f};
  bool active_{false};
  double last_blocked_{-1e9};
  double side_lost_since_{-1.0};
};

}

#endif
