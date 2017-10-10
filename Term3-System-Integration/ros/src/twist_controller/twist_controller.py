from pid import PID
from yaw_controller import YawController
from math import exp, expm1
from lowpass import LowPassFilter
import rospy


GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self, ks_pos, ks_vel, data):
        # TODO: Implement
        decel_limit = rospy.get_param('~decel_limit', -5)
        accel_limit = rospy.get_param('~accel_limit', 1.)
        max_steer_angle = rospy.get_param('~max_steer_angle', 8.)

        
        kp_pos, ki_pos, kd_pos = ks_pos[0],ks_pos[1], ks_pos[2]
        kp_vel, ki_vel, kd_vel = ks_vel[0], ks_vel[1], ks_vel[2]

        self.pid_steer = PID(kp_pos, ki_pos, kd_pos,-max_steer_angle,max_steer_angle ) 
        self.pid_vel = PID(kp_vel, ki_vel, kd_vel, decel_limit, accel_limit) 

        self.filter_velocity = LowPassFilter(0.8,0.2)
        self.filter_steering = LowPassFilter(0.5,0.5)

        wheel_base = data[0]
        steer_ratio = 1.5*data[1]
        min_speed = data[2]
        max_lat_accel = data[3]
        max_steer_angl = data[4]

        self.steering = YawController(wheel_base, steer_ratio, min_speed, max_lat_accel, max_steer_angl)


    def control(self, desired_linear_velocity, desired_angular_velocity, current_velocity, sample_time, dbw_enable):
        # TODO: Change the arg, kwarg list to suit your needs
        vehicle_mass = rospy.get_param('~vehicle_mass', 1736.35)
        fuel_capacity = rospy.get_param('~fuel_capacity', 13.5)
        brake_deadband = rospy.get_param('~brake_deadband', .1)
        wheel_radius = rospy.get_param('~wheel_radius', 0.2413)
        decel_limit = rospy.get_param('~decel_limit', -5)

        delta = sample_time.to_sec()
        pid_steer_enable = False
        current_linear_velocity = current_velocity.linear.x
        error_vel = desired_linear_velocity - current_linear_velocity

        
        angular_velocity = current_velocity.angular.z
        error_steer = desired_angular_velocity - angular_velocity

        #error_steer = 0 
        if not dbw_enable:
        	self.pid_steer.reset()
        	self.pid_vel.reset()

        if pid_steer_enable:
            steer = self.pid_steer.step(error_steer, delta)
        else:
        	steering = self.filter_steering.filt(desired_angular_velocity)
        	steer = self.filter_steering.get()
        	steer = self.steering.get_steering(desired_linear_velocity, steer, current_linear_velocity)

        brake = 0 
        throttle = 0
        acceleration = self.pid_vel.step(error_vel, delta)
        filter_acc = self.filter_velocity.filt(acceleration)


        
        if filter_acc:                
                if filter_acc > 0.0:
                	throttle = filter_acc          

                else:
                	#Should be -acceleration?
                	# brake = 20000
                    # brake = (20000/decel_limit) * filter_acc
                    brake = -(vehicle_mass + fuel_capacity*GAS_DENSITY) * filter_acc * wheel_radius

                    # brake = (vehicle_mass)

        return throttle, brake, steer
