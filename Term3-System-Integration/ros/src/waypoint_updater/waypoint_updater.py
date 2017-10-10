#!/usr/bin/env python
import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane  # Waypoint import not used
from std_msgs.msg import Int32

import tf
import math
# import time -- only used for uncommenting logging

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 50 # Number of waypoints we will publish. You can change this number
ONE_MPH = 0.44704


def dl (a, b):
    return math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)


class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)
        # rospy.Subscriber('/obstacle_waypoint', Int32, self.obstacle_cb)

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)
        self.current_position = None
        self.traffic_light = -10
        self.obstacle = None
        self.velocity = rospy.get_param('~velocity')*ONE_MPH

        self.loop()

    def loop(self):

        rate = rospy.Rate(2)

        while not rospy.is_shutdown():
            # time0 = time.time()

            if self.current_position is not None and self.waypoints is not None:
                waypoints = self.waypoints.waypoints
                next_wp_index = self.next_waypoint(self.current_position.pose, waypoints)
                print("Waypoint:", next_wp_index, "Traffic:", self.traffic_light, "Diff:",self.traffic_light-next_wp_index)
                # print()
                ENDPOINT = next_wp_index + LOOKAHEAD_WPS
                END_OF_MAP = len(waypoints)
                # ENDPOINT = min(next_wp_index + LOOKAHEAD_WPS, len(waypoints))
                # rospy.loginfo("the endpoint: %f", ENDPOINT)
                final_waypoints_msg = Lane()
                final_waypoints_msg.header.frame_id = '/world'
                final_waypoints_msg.header.stamp = rospy.Time(0)
                # braking_zone = min(next_wp_index + 10,  len(waypoints))

                braking_zone = next_wp_index + 10
                if (self.traffic_light > next_wp_index and self.traffic_light < braking_zone):
                    # now = rospy.get_rostime()
                    # print("here")
                    # rospy.loginfo("Current time %i %i", now.secs, now.nsecs)
                    my_waypoints = waypoints[next_wp_index:braking_zone]
                    for i in range(len(my_waypoints)):
                        self.set_waypoint_velocity(my_waypoints, i, 0)

                elif (self.traffic_light > braking_zone and self.traffic_light < ENDPOINT):
                    my_waypoints = waypoints[next_wp_index:ENDPOINT]
                    for i in range(len(my_waypoints)):
                        self.set_waypoint_velocity(my_waypoints, i, 5.0)

                elif (ENDPOINT>END_OF_MAP):
                    my_waypoints = waypoints[next_wp_index:END_OF_MAP]
                    for i in range(len(my_waypoints)):
                        self.set_waypoint_velocity(my_waypoints, i, self.velocity)

                    # setting last 10 point to 0
                    for j in range(10):
                        self.set_waypoint_velocity(my_waypoints, len(my_waypoints)-1-j, 0.)

                else:
                    # now = rospy.get_rostime()
                    # rospy.loginfo("Current time %i %i", now.secs, now.nsecs)
                    my_waypoints = waypoints[next_wp_index:ENDPOINT]

                    for i in range(len(my_waypoints)):
                        self.set_waypoint_velocity(my_waypoints, i, self.velocity)
                final_waypoints_msg.waypoints = my_waypoints
                # create and publish the waypoints in front of the car

                # time1 = time.time() -- only used in logging, uncomment
                # new_time = (time1-time0)*1000. -- only uncomment if logged
                # rospy.loginfo("Time it took to publish waypoints: %f", new_time)
                self.final_waypoints_pub.publish(final_waypoints_msg)
            rate.sleep()

    def pose_cb(self, msg):
        self.current_position = msg

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints

    def traffic_cb(self, msg):
        # Callback for /traffic_waypoint message.
        self.traffic_light = msg.data

    def obstacle_cb(self, msg):
        # Callback for /obstacle_waypoint message.
        self.obstacle = msg.data

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist

    def closest_waypoint(self, pose, waypoints):
        return min([(i, dl(pose.position, wp.pose.pose.position)) for i, wp in enumerate(waypoints)],
                   key=lambda x: x[1])[0]

    def next_waypoint(self, pose, waypoints):

        next_wp_index = self.closest_waypoint(pose, waypoints)
        wp_pose = waypoints[next_wp_index].pose.pose

        heading = math.atan2((wp_pose.position.y - pose.position.y), (wp_pose.position.x - pose.position.x))

        pose_quaternion = (pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w)
        (roll, pitch, yaw) = tf.transformations.euler_from_quaternion(pose_quaternion)
        angle = math.fabs(heading - yaw)

        if angle > (math.pi / 4):
            next_wp_index += 1

        return next_wp_index


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
