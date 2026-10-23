#ifndef MPPI_CRITICS__SIDE_CHOICE_CRITIC_HPP_
#define MPPI_CRITICS__SIDE_CHOICE_CRITIC_HPP_

#include "nav2_mppi_controller/critic_function.hpp"

namespace mppi::critics
{

class SideChoiceCritic : public mppi::critics::CriticFunction
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
  float forward_weight_{10.0f};
  float min_rotation_speed_{0.20f};
  float time_window_{1.0f};
  float check_distance_{2.0f};
  float min_distance_{0.5f};
  int block_cost_{253};
  int free_cost_{128};
  float max_offset_{2.0f};
  float min_gap_{0.7f};
  float width_cap_{2.0f};
  float path_half_length_{0.4f};
  float space_weight_{1.0f};
  float path_weight_{1.0f};
  float activation_angle_deg_{20.0f};
  float deactivation_angle_deg_{8.0f};
  float release_time_{2.0f};
  float side_{0.0f};
  bool steering_{false};
  double last_blocked_{-1e9};
};

}

#endif
