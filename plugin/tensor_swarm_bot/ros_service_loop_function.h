#ifndef PROJECT_ROSSERVICELOOPFUNCTION_H
#define PROJECT_ROSSERVICELOOPFUNCTION_H

// Copyright (C) 2018 deeplearningrobotics.ai
#include <argos3/core/simulator/loop_functions.h>
#include <ros/ros.h>
#include <thread>
#include <mutex>
#include <condition_variable>

#include "tensorswarm/AIService.h"

class ROSServiceLoopFunction : public argos::CLoopFunctions {

public:
  ROSServiceLoopFunction();

  virtual ~ROSServiceLoopFunction() = default;

  virtual void Init(argos::TConfigurationNode &t_tree) override;

  /// Callback for the ROS service.
  bool ServiceFunction(const tensorswarm::AIServiceRequest & req,
                       tensorswarm::AIServiceResponse &resp);

  virtual void Reset() override;

  virtual void PreStep() override;


  virtual void PostStep() override;

private:
  std::mutex m_m_main;
  std::condition_variable m_cv_main;
  std::condition_variable m_cv_service;
  std::thread m_ros_thread;
  bool m_service_data_available;
  bool m_loop_done;
  tensorswarm::AIServiceRequest m_req_store;
  tensorswarm::AIServiceResponse m_resp_store;
  std::unique_lock<std::mutex> m_lk;
  uint m_episode_time;
};


#endif //PROJECT_ROSSERVICELOOPFUNCTION_H
