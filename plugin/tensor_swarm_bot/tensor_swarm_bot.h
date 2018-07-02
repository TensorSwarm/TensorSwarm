#ifndef tensorswarm_BOT_H
#define tensorswarm_BOT_H

// Copyright (C) 2018 deeplearningrobotics.ai
#include <string>

#include <argos3/core/control_interface/ci_controller.h>
#include <argos3/plugins/robots/generic/control_interface/ci_differential_steering_actuator.h>
#include <argos3/plugins/robots/generic/control_interface/ci_differential_steering_sensor.h>
#include <argos3/plugins/robots/foot-bot/control_interface/ci_footbot_lidar_sensor.h>
#include <argos3/core/utility/math/vector3.h>

#include <geometry_msgs/Twist.h>
#include <geometry_msgs/Pose2D.h>

#include "tensorswarm/AIService.h"

using namespace argos;

class CTensorSwarmBot : public CCI_Controller {

public:

  CTensorSwarmBot();
  virtual ~CTensorSwarmBot() = default;

  void Init(TConfigurationNode& t_node) override;

  void ControlStep() override;

  void Reset() override;

  /// Returns the scan from the robots laser scanner.
  tensorswarm::LaserScan getLaser() const;

  /// Returns the robots velocity.
  geometry_msgs::Twist getVelocities() const;

  /// Sets the robots velocity.
  void setVelocity(const geometry_msgs::Twist& twist);

  /// Sets a new goal for the robot.
  void setNewGoal(const geometry_msgs::Pose2D& new_goal);

  /// Gets the goal of the robot.
  geometry_msgs::Pose2D getGoal() const;

  /// Returns the distance to the robots goal given its \param currentPosition.
  double goalDistance(const argos::CVector3& currentPosition) const;

  /// Returns the distance to the robots goal given its \param currentPosition.
  double goalDistance(const geometry_msgs::Pose2D &currentPosition) const;

  /**   Provided its \param currentPosition returns the robots progress towards
   *    the goal during the lats timestep. Do not call this function multiple times
   *    in an iteration!
   */
  double goalProgress(const argos::CVector3& currentPosition);

  bool arrived() {return m_arrived;}
  void setArrived() {m_arrived = true;}

private:

  CCI_DifferentialSteeringActuator* m_pcWheels;
  CCI_DifferentialSteeringSensor* m_pcWheelsSensor;
  CCI_FootBotLidarSensor* m_pcLaser;

  static constexpr const Real HALF_BASELINE = 0.07f;
  static constexpr const Real WHEEL_RADIUS = 0.029112741f;


  geometry_msgs::Pose2D m_goal;
  double m_previous_goal_distance;
  bool m_goal_progress_called;
  bool m_arrived = false;
};

#endif
