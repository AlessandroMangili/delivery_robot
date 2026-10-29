#include <cmath>
#include <string>

#include "behaviortree_cpp_v3/bt_factory.h"
#include "behaviortree_cpp_v3/control_node.h"
#include "nav_msgs/msg/path.hpp"
#include "rclcpp/rclcpp.hpp"

namespace delivery_bt_nodes
{

class ProgressRecoveryNode : public BT::ControlNode
{
public:
  ProgressRecoveryNode(const std::string & name, const BT::NodeConfiguration & conf)
  : BT::ControlNode(name, conf), logger_(rclcpp::get_logger("ProgressRecoveryNode"))
  {
    getInput("number_of_retries", number_of_retries_);
    getInput("progress_distance", progress_distance_);
  }

  static BT::PortsList providedPorts()
  {
    return {
      BT::InputPort<int>("number_of_retries", 1, "Number of retries"),
      BT::InputPort<double>("progress_distance", 1.0,
        "Drop of the remaining path length that gives back all the retries"),
      BT::InputPort<nav_msgs::msg::Path>("path", "Current global path")};
  }

  BT::NodeStatus tick() override
  {
    if (children_nodes_.size() != 2) {
      throw BT::BehaviorTreeException(
              "Recovery Node '" + name() + "' must only have 2 children.");
    }
    setStatus(BT::NodeStatus::RUNNING);
    while (current_child_idx_ < 2 && retry_count_ <= number_of_retries_) {
      TreeNode * child_node = children_nodes_[current_child_idx_];
      const BT::NodeStatus child_status = child_node->executeTick();
      if (current_child_idx_ == 0) {
        switch (child_status) {
          case BT::NodeStatus::SUCCESS:
            halt();
            return BT::NodeStatus::SUCCESS;
          case BT::NodeStatus::FAILURE:
            if (retry_count_ < number_of_retries_) {
              reference_ = remaining();
              ControlNode::haltChild(0);
              current_child_idx_++;
              break;
            }
            RCLCPP_WARN(
              logger_, "%s: %d recoveries used without %.1f m of progress, giving up",
              name().c_str(), retry_count_, progress_distance_);
            halt();
            return BT::NodeStatus::FAILURE;
          case BT::NodeStatus::RUNNING:
            checkProgress();
            return BT::NodeStatus::RUNNING;
          default:
            throw BT::LogicError("A child node must never return IDLE");
        }
      } else {
        switch (child_status) {
          case BT::NodeStatus::SUCCESS:
            ControlNode::haltChild(1);
            retry_count_++;
            current_child_idx_--;
            break;
          case BT::NodeStatus::FAILURE:
            halt();
            return BT::NodeStatus::FAILURE;
          case BT::NodeStatus::RUNNING:
            return BT::NodeStatus::RUNNING;
          default:
            throw BT::LogicError("A child node must never return IDLE");
        }
      }
    }
    halt();
    return BT::NodeStatus::FAILURE;
  }

  void halt() override
  {
    ControlNode::halt();
    retry_count_ = 0;
    current_child_idx_ = 0;
    reference_ = -1.0;
  }

private:
  double remaining()
  {
    nav_msgs::msg::Path path;
    if (!getInput("path", path) || path.poses.size() < 2) {
      return -1.0;
    }
    double length = 0.0;
    for (size_t i = 1; i < path.poses.size(); ++i) {
      const auto & a = path.poses[i - 1].pose.position;
      const auto & b = path.poses[i].pose.position;
      length += std::hypot(b.x - a.x, b.y - a.y);
    }
    return length;
  }

  void checkProgress()
  {
    if (retry_count_ == 0 || reference_ < 0.0) {
      return;
    }
    const double now = remaining();
    if (now >= 0.0 && now <= reference_ - progress_distance_) {
      RCLCPP_INFO(
        logger_, "\033[32m%s: %.1f m of progress since the last recovery, %d recoveries given back\033[0m",
        name().c_str(), reference_ - now, retry_count_);
      retry_count_ = 0;
      reference_ = -1.0;
    }
  }

  rclcpp::Logger logger_;
  unsigned int current_child_idx_{0};
  int number_of_retries_{1};
  int retry_count_{0};
  double progress_distance_{1.0};
  double reference_{-1.0};
};

}

BT_REGISTER_NODES(factory)
{
  factory.registerNodeType<delivery_bt_nodes::ProgressRecoveryNode>("ProgressRecoveryNode");
}
