#ifndef MPPI_CRITICS__ROTATION_DEADZONE_CRITIC_HPP_
#define MPPI_CRITICS__ROTATION_DEADZONE_CRITIC_HPP_

#include "nav2_mppi_controller/critic_function.hpp"

namespace mppi::critics
{

class RotationDeadzoneCritic : public mppi::critics::CriticFunction
{
public:
  void initialize() override;
  void score(mppi::CriticData & data) override;

protected:
  unsigned int power_{1};
  float weight_{10.0f};
  float v_max_{0.08f};
  float w_min_{0.06f};
  float stall_w_{0.19f};
  float window_s_{1.0f};
  float brake_v_{0.0f};
};

}

#endif
