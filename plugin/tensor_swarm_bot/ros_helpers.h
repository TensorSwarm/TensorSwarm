#ifndef PROJECT_ROS_HELPERS_H
#define PROJECT_ROS_HELPERS_H

// Copyright (C) 2018 deeplearningrobotics.ai
#include <argos3/core/utility/math/vector3.h>
#include <argos3/core/utility/math/quaternion.h>
#include <geometry_msgs/Pose2D.h>

/// Converts \param pose to an argos::CVector3.
inline argos::CVector3 convertVec(const geometry_msgs::Pose2D& pose) {
  return argos::CVector3(pose.x, pose.y, 0.0);
}

/// Converts \param pose to an argos::CQuaternion.
inline argos::CQuaternion convertQuat(const geometry_msgs::Pose2D& pose) {
  argos::CQuaternion quat;
  const argos::CVector3 axis(0.0, 0.0, 1.0);
  const argos::CRadians radians(pose.theta);
  quat.FromAngleAxis(radians, axis);
  return quat;
}

#endif //PROJECT_ROS_HELPERS_H
