// Rilancia /clock a frequenza ridotta per i nodi Python.
//
// Gazebo avanza a passi di 1 ms e pubblica /clock a ogni passo (~870 msg/s
// reali). Ogni messaggio risveglia l'executor di rclpy: un nodo Python vuoto
// con use_sim_time consuma ~46% di un core solo per questo. I nodi Python
// della pipeline si remappano su /clock_throttled; i nodi C++ (controller,
// localizzazione) restano su /clock.

#include <memory>

#include "rclcpp/rclcpp.hpp"
#include "rosgraph_msgs/msg/clock.hpp"

class ClockThrottle : public rclcpp::Node
{
public:
  ClockThrottle()
  : Node("clock_throttle")
  {
    const double period_s = declare_parameter<double>("period_s", 0.01);
    period_ns_ = static_cast<int64_t>(period_s * 1e9);
    // publisher RELIABLE: compatibile sia con i subscriber best-effort (rclpy
    // TimeSource) sia con quelli reliable
    pub_ = create_publisher<rosgraph_msgs::msg::Clock>(
      declare_parameter<std::string>("output_topic", "/clock_throttled"),
      rclcpp::QoS(1).reliable());
    sub_ = create_subscription<rosgraph_msgs::msg::Clock>(
      declare_parameter<std::string>("input_topic", "/clock"), rclcpp::ClockQoS(),
      [this](rosgraph_msgs::msg::Clock::ConstSharedPtr msg) {
        const int64_t t = rclcpp::Time(msg->clock).nanoseconds();
        // t < last_: la simulazione e' ripartita, si inoltra subito
        if (last_ns_ < 0 || t < last_ns_ || t - last_ns_ >= period_ns_) {
          pub_->publish(*msg);
          last_ns_ = t;
        }
      });
    RCLCPP_INFO(get_logger(), "clock_throttle: periodo %.3f s", period_s);
  }

private:
  int64_t period_ns_{10000000};
  int64_t last_ns_{-1};
  rclcpp::Publisher<rosgraph_msgs::msg::Clock>::SharedPtr pub_;
  rclcpp::Subscription<rosgraph_msgs::msg::Clock>::SharedPtr sub_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ClockThrottle>());
  rclcpp::shutdown();
  return 0;
}
