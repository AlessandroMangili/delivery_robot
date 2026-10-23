#ifndef MPPI_CRITICS__CLEARANCE_CRITIC_HPP_
#define MPPI_CRITICS__CLEARANCE_CRITIC_HPP_

#include <utility>
#include <vector>

#include "nav2_mppi_controller/critic_function.hpp"

namespace mppi::critics
{

class ClearanceCritic : public mppi::critics::CriticFunction
{
public:
  void initialize() override;
  void score(mppi::CriticData & data) override;

protected:
  void updateDistanceField();
  float distanceAt(float wx, float wy) const;

  float weight_{30.0f};
  float safe_distance_{0.30f};
  float perimeter_step_{0.25f};
  double last_field_time_{-1.0};
  int trajectory_point_step_{3};
  int obstacle_cost_{254};
  std::vector<float> field_;
  unsigned int size_x_{0};
  unsigned int size_y_{0};
  double resolution_{0.05};
  double origin_x_{0.0};
  double origin_y_{0.0};
  std::vector<std::pair<float, float>> perimeter_;
};

}

#endif
