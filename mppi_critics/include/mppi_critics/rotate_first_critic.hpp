#ifndef MPPI_CRITICS__ROTATE_FIRST_CRITIC_HPP_
#define MPPI_CRITICS__ROTATE_FIRST_CRITIC_HPP_

#include "nav2_mppi_controller/critic_function.hpp"

namespace mppi::critics
{

class RotateFirstCritic : public mppi::critics::CriticFunction
{
public:
  void initialize() override;
  void score(mppi::CriticData & data) override;

protected:
  float rotation_weight_{10.0f};
  float forward_weight_{20.0f};
  float min_rotation_speed_{0.20f};
  float time_window_{1.0f};
  float activation_angle_deg_{45.0f};
  float deactivation_angle_deg_{15.0f};
  float heading_lookahead_{1.0f};
  float goal_distance_{0.5f};
  float block_check_distance_{0.0f};
  int block_cost_threshold_{254};
  bool active_{false};
};

}

#endif
