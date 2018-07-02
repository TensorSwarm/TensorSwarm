source /opt/ros/kinetic/setup.bash
source devel/setup.bash
export ARGOS_PLUGIN_PATH=$ARGOS_PLUGIN_PATH:./devel/lib

roscore &

catkin_make && argos3 -c src/tensorswarm/argos_worlds/4_way.argos &
argos3 -c src/tensorswarm/argos_worlds/4_way.argos &
argos3 -c src/tensorswarm/argos_worlds/4_way.argos &
argos3 -c src/tensorswarm/argos_worlds/4_way.argos &

python2 src/tensorswarm/scripts/new/RunExperiment4Way.py