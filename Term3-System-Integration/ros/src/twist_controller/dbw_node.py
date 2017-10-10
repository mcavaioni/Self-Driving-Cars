#!/usr/bin/env python

import rospy
from std_msgs.msg import Bool
from dbw_mkz_msgs.msg import ThrottleCmd, SteeringCmd, BrakeCmd, SteeringReport
from geometry_msgs.msg import TwistStamped
from styx_msgs.msg import Lane, Waypoint
import math
from geometry_msgs.msg import PoseStamped
from twist_controller import Controller

'''
You can build this node only after you have built (or partially built) the `waypoint_updater` node.

You will subscribe to `/twist_cmd` message which provides the proposed linear and angular velocities.
You can subscribe to any other message that you find important or refer to the document for list
of messages subscribed to by the reference implementation of this node.

One thing to keep in mind while building this node and the `twist_controller` class is the status
of `dbw_enabled`. While in the simulator, its enabled all the time, in the real car, that will
not be the case. This may cause your PID controller to accumulate error because the car could
temporarily be driven by a human instead of your controller.

We have provided two launch files with this node. Vehicle specific values (like vehicle_mass,
wheel_base) etc should not be altered in these files.

We have also provided some reference implementations for PID controller and other utility classes.
You are free to use them or build your own.

Once you have the proposed throttle, brake, and steer values, publish it on the various publishers
that we have created in the `__init__` function.

'''

class DBWNode(object):
	def __init__(self):
		rospy.init_node('dbw_node', log_level=rospy.INFO)

		
		vehicle_mass = rospy.get_param('~vehicle_mass', 1736.35)
		fuel_capacity = rospy.get_param('~fuel_capacity', 13.5)
		brake_deadband = rospy.get_param('~brake_deadband', .1)
		decel_limit = rospy.get_param('~decel_limit', -5)
		accel_limit = rospy.get_param('~accel_limit', 1.)
		wheel_radius = rospy.get_param('~wheel_radius', 0.2413)
		wheel_base = rospy.get_param('~wheel_base', 2.8498)
		steer_ratio = rospy.get_param('~steer_ratio', 14.8)
		max_lat_accel = rospy.get_param('~max_lat_accel', 3.)
		max_steer_angle = rospy.get_param('~max_steer_angle', 8.)
		min_speed = 0

		self.steer_pub = rospy.Publisher('/vehicle/steering_cmd',
										 SteeringCmd, queue_size=1)
		self.throttle_pub = rospy.Publisher('/vehicle/throttle_cmd',
											ThrottleCmd, queue_size=1)
		self.brake_pub = rospy.Publisher('/vehicle/brake_cmd',
										 BrakeCmd, queue_size=1)
		self.dbw_enabled = False
		self.current_velocity = None
		self.waypoints_x = None
		self.waypoints_y = None
		self.pose_x = None
		self.pose_y = None

		self.previous_time = rospy.get_rostime()

		# TODO: Create `TwistController` object
		kp_steer, ki_steer, kd_steer = 1.5,0.005,0.7
		#kp_vel, ki_vel, kd_vel = 0.2,0.005,0.1
		kp_vel, ki_vel, kd_vel = 10.0,0.0,0.0

		self.controller = Controller( (kp_steer, ki_steer, kd_steer), (kp_vel, ki_vel, kd_vel), (wheel_base, steer_ratio, decel_limit, max_lat_accel, max_steer_angle))

		# TODO: Subscribe to all the topics you need to
		rospy.Subscriber("/vehicle/dbw_enabled", Bool, self.dbw_enabled_cb)
		rospy.Subscriber("/current_velocity", TwistStamped, self.current_velocity_cb)

		#to get final waypoints from waypoint updater. we cannot suscribe until node is ready. Should we obtained linear and angular from /final_waypoints 
		#or /twist_cmd
		rospy.Subscriber("/twist_cmd", TwistStamped, self.twist_cmd_cb)
		rospy.Subscriber("/final_waypoints", Lane, self.get_waypoints_cb)
		rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)


  

		self.loop()

	def dbw_enabled_cb(self, msg):
			self.dbw_enabled = msg.data

	def current_velocity_cb(self, msg):
			#should get linear velocity and angular velocity
			self.current_velocity = msg.twist
			self.time = msg.header.stamp
			
			#print linear velocity
			#rospy.loginfo("Current linear velocity: %i", self.current_velocity.linear.x)
			#print angular velocity
			#rospy.loginfo("Current angular velocity: %i", self.current_velocity.angular.x)

	def twist_cmd_cb(self, msg):
			self.twist_cmd = msg.twist
			

	def get_waypoints_cb(self, waypoints):
			self.waypoints_x = waypoints.waypoints[0].pose.pose.position.x
			self.waypoints_y = waypoints.waypoints[0].pose.pose.position.y

	def pose_cb(self, msg):
        # TODO: Implement
	        self.pose_x = msg.pose.position.x
	        self.pose_y = msg.pose.position.y
	        self.pose_z = msg.pose.position.z
	        self.orient_x = msg.pose.orientation.x
	        self.orient_y = msg.pose.orientation.y
	        self.orient_z = msg.pose.orientation.z
	        self.orient_w = msg.pose.orientation.w
				

	def loop(self):
		rate = rospy.Rate(50) # 50Hz
		while not rospy.is_shutdown():
			# TODO: Get predicted throttle, brake, and steering using `twist_controller`
			# You should only publish the control commands if dbw is enabled
			# throttle, brake, steering = self.controller.control(<proposed linear velocity>,
			#                                                     <proposed angular velocity>,
			#                                                     <current linear velocity>,
			#                                                     <dbw status>,
			#                                                     <any other argument you need>)

			
			if self.dbw_enabled: 
				#rospy.loginfo("twist_cmd_linear: %i",self.twist_cmd.linear.x)
				#rospy.loginfo("twist_cmd_angular: %i",self.twist_cmd.angular.x)


				if self.current_velocity is not None:

					#rospy.loginfo("%f %f %f %f", self.waypoints_x, self.waypoints_y, self.pose_x, self.pose_y)

					#rospy.loginfo("current_velocity_linear: %i",self.current_velocity.linear.x)
					now = rospy.get_rostime()
					sample_time = now - self.previous_time
					#rospy.loginfo("Now time: %f Previous: %f  Sample time: %f curr: %f", now.nsecs, self.previous_time.nsecs, sample_time.nsecs, self.time.nsecs)

					self.previous_time = now
					desired_linear_velocity = self.twist_cmd.linear.x
					desired_angular_velocity = self.twist_cmd.angular.z
					#rospy.loginfo("Linear_velocity_des: %f ",desired_linear_velocity)
					#rospy.loginfo('des_lin_vel:' + str(desired_linear_velocity) + '\t' + 'des_ang_vel: ' + str(desired_angular_velocity) +'\t'+ 'curr_lin_vel: ' + str(self.current_velocity.linear.x) + '\t' + 'des_ang_vel: ' + str(self.current_velocity.angular.z))
					
					throttle, brake, steering = self.controller.control(desired_linear_velocity, desired_angular_velocity, self.current_velocity, sample_time, self.dbw_enabled)
					#Angular_velocity_des: %f Angular_velocy_curr: %f Steering: %f Line_velocy_curr: %f Line_velocy_des: %f
					#rospy.loginfo("%f %f %f %f %f %f %f %f %f", self.waypoints_x, self.waypoints_y, self.pose_x, self.pose_y, desired_angular_velocity, self.current_velocity.angular.z,steering, desired_linear_velocity, self.current_velocity.linear.x)
					# throttle = 0.5                
					# brake = 0
					# steer = 0.5
					self.publish(throttle, brake, steering)
					# self.publish(throttle, steering)
			rate.sleep()

	def publish(self, throttle, brake, steer):
		tcmd = ThrottleCmd()
		tcmd.enable = True
		tcmd.pedal_cmd_type = ThrottleCmd.CMD_PERCENT
		tcmd.pedal_cmd = throttle
		self.throttle_pub.publish(tcmd)

		scmd = SteeringCmd()
		scmd.enable = True
		scmd.steering_wheel_angle_cmd = steer
		self.steer_pub.publish(scmd)

		bcmd = BrakeCmd()
		bcmd.enable = True
		bcmd.pedal_cmd_type = BrakeCmd.CMD_TORQUE
		bcmd.pedal_cmd = brake
		self.brake_pub.publish(bcmd)



if __name__ == '__main__':
	DBWNode()
