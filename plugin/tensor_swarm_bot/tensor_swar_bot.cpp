// Copyright (C) 2018 deeplearningrobotics.ai
#include "tensor_swarm_bot.h"

#include <argos3/core/utility/configuration/argos_configuration.h>
#include <argos3/core/utility/math/vector2.h>

#include "ros_helpers.h"


using namespace tensorswarm;

CTensorSwarmBot::CTensorSwarmBot() :
  m_pcWheels(nullptr),
  m_pcWheelsSensor(nullptr),
  m_pcLaser(nullptr),
  m_previous_goal_distance(-1.0),
  m_goal_progress_called(false)
{}

void CTensorSwarmBot::Init(TConfigurationNode& t_node) {
  m_pcWheels = GetActuator<CCI_DifferentialSteeringActuator>("differential_steering");
  m_pcWheelsSensor = GetSensor<CCI_DifferentialSteeringSensor>("differential_steering");
  m_pcLaser = GetSensor<CCI_FootBotLidarSensor>("footbot_lidar");
}

void CTensorSwarmBot::ControlStep() {
  m_goal_progress_called = false;
}

tensorswarm::LaserScan CTensorSwarmBot::getLaser() const
{
  const CCI_FootBotLidarSensor::TReadings& laserReadings = m_pcLaser->GetReadings();
  LaserScan laserScan;

  for (const auto& reads : laserReadings) {
    LaserRay laserRay;
    laserRay.value = reads.Value;
    laserRay.angle = reads.Angle.GetValue();
    laserScan.laser_rays.push_back(laserRay);
  }
  return laserScan;
}

geometry_msgs::Twist CTensorSwarmBot::getVelocities() const
{
  geometry_msgs::Twist twist;
  const CCI_DifferentialSteeringSensor::SReading& readings = m_pcWheelsSensor->GetReading();
  twist.linear.x = (readings.VelocityLeftWheel + readings.VelocityRightWheel)*WHEEL_RADIUS / 2.0;
  twist.angular.z = (readings.VelocityRightWheel - readings.VelocityLeftWheel)*WHEEL_RADIUS / (2.0*HALF_BASELINE);
  return twist;
}

void CTensorSwarmBot::setVelocity(const geometry_msgs::Twist& twist) {
  Real v = twist.linear.x;
  Real w = twist.angular.z;

  const auto leftSpeed = (v - HALF_BASELINE * w) / WHEEL_RADIUS;
  const auto rightSpeed = (v + HALF_BASELINE * w) / WHEEL_RADIUS;

  m_pcWheels->SetLinearVelocity(leftSpeed, rightSpeed);
}

void CTensorSwarmBot::setNewGoal(const geometry_msgs::Pose2D &new_goal) {
  m_goal = new_goal;
  m_previous_goal_distance = -1.0;
}

geometry_msgs::Pose2D CTensorSwarmBot::getGoal() const {
  return m_goal;
}
double CTensorSwarmBot::goalDistance(const argos::CVector3& currentPosition) const{
  return (currentPosition - convertVec(m_goal)).Length();
}

double CTensorSwarmBot::goalDistance(const geometry_msgs::Pose2D& currentPosition) const {
 return (convertVec(currentPosition) - convertVec(m_goal)).Length();
}

double CTensorSwarmBot::goalProgress(const argos::CVector3& currentPosition) {
  if(m_goal_progress_called) {
    throw std::logic_error("You are not allowed to call goalProgress() twice in an iteration.");
  }
  m_goal_progress_called = true;

  double result = 0.0;
  if(m_previous_goal_distance >= 0.0) {
    result = m_previous_goal_distance - goalDistance(currentPosition);
  }
  m_previous_goal_distance = goalDistance(currentPosition);
  if(std::fabs(result) > 0.2) {
    throw std::logic_error("Goal progress to large!");
  }
  return result;
}

void CTensorSwarmBot::Reset() {
  m_arrived = false;
  m_previous_goal_distance = -1.0;
}

REGISTER_CONTROLLER(CTensorSwarmBot, "tensor_swarm_bot_controller")
