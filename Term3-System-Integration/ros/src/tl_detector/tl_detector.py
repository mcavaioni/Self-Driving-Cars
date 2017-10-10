#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import yaml
import math
import os
import sys
from functools import partial

def dl(a, b):
    """Duplicates `dl` function in waypoint_updater for now (to avoid
    common import)."""
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)

STATE_COUNT_THRESHOLD = 3
STATE_ENUM = {0: "RED",
              1: "YELLOW",
              2: "GREEN"}

def join_chunks(directory, filename, chunksize=1024):
    print "restoring model:", filename, "from directory:", directory
    if os.path.exists(directory):
        if os.path.exists(filename):
            os.remove(filename)
        output = open(filename, 'wb')
        chunks = os.listdir(directory)
        chunks.sort()
        for fname in chunks:
            fpath = os.path.join(directory, fname)
            with open(fpath, 'rb') as fileobj:
                for chunk in iter(partial(fileobj.read, chunksize), ''):
                    output.write(chunk)
        output.close()

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.stop_lights_pos = []

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        config_string = rospy.get_param("/traffic_light_config")

        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
        # self.light_classifier = TLClassifier()

        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0
        self.lights_cache = {}
        self.firstLight = None
        self.firstLightIndex = None
        self.load_stop_point()

        model_ckpt = rospy.get_param('~model_ckpt')
        directory, _ = os.path.splitext(model_ckpt)
        print(directory)
        join_chunks(directory, model_ckpt)
        self.light_classifier = TLClassifier(model_ckpt)
        rate = rospy.Rate(1000)
        rospy.spin()

    def load_stop_point(self):
        stop_line_positions = self.config['stop_line_positions']

        for stop_line_position in stop_line_positions:
            stop_line_position_pose = Pose()
            stop_line_position_pose.position.x = stop_line_position[0]
            stop_line_position_pose.position.y = stop_line_position[1]
            stop_line_position_pose.position.z = 0
            self.stop_lights_pos.append(stop_line_position_pose)
            # self.tl_waypoints_idx.append(
            #     self.get_closest_waypoint_idx(stop_line_position_pose)
            # )

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints

    def traffic_cb(self, msg):
        pass
        # if len(self.lights)==0:
        #     self.lights = msg.lights
        #     self.populate_cache()

    def populate_cache(self):

        for light in self.stop_lights_pos:
            if light not in self.lights_cache:
                self.lights_cache[light] = self.get_closest_waypoint(light)

        self.firstLight, self.firstLightIndex = self.get_first_traffic_light(self.stop_lights_pos)

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.has_image = True
        self.camera_image = msg
        light_wp, state = self.process_traffic_lights()

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if self.state != state:
            self.state_count = 0
            self.state = state


        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            if (state == TrafficLight.RED):
                light_wp = light_wp

            else:
                light_wp = -1

            self.last_wp = light_wp
            if self.should_publish(state):
                print("publishing traffic light ... ", self.get_color(state))
                self.upcoming_red_light_pub.publish(Int32(light_wp))
                now = rospy.get_rostime()
                # rospy.loginfo("Current time %i %i", now.secs, now.nsecs)
        else:
            if self.should_publish(state):
                print("publishing traffic light state... ", self.get_color(state))
                self.upcoming_red_light_pub.publish(Int32(self.last_wp))
                # now = rospy.get_rostime()
                # rospy.loginfo("Current time %i %i", now.secs, now.nsecs)
        self.state_count += 1

    def should_publish(self, state):
        return state == TrafficLight.RED or state == TrafficLight.YELLOW or state == TrafficLight.GREEN

    def get_color(self, state):
        return STATE_ENUM.get(state, "UNKNOWN")

    def get_closest_waypoint(self, pose):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints
        """
        if self.waypoints is None:
            return 0
        return min([(i, dl(pose.position, wp.pose.pose.position)) for i, wp in enumerate(self.waypoints.waypoints)],
                   key=lambda x: x[1])[0]

    def project_to_image_plane(self, point_in_world):
        """Project point from 3D world coordinates to 2D camera image location

        Args:
            point_in_world (Point): 3D location of a point in the world

        Returns:
            x (int): x coordinate of target point in image
            y (int): y coordinate of target point in image

        """

        fx = self.config['camera_info']['focal_length_x']
        fy = self.config['camera_info']['focal_length_y']
        image_width = self.config['camera_info']['image_width']
        image_height = self.config['camera_info']['image_height']

        # get transform between pose of camera and world frame
        trans = None
        try:
            now = rospy.Time.now()
            self.listener.waitForTransform("/base_link",
                  "/world", now, rospy.Duration(1.0))
            (trans, rot) = self.listener.lookupTransform("/base_link",
                  "/world", now)

        except (tf.Exception, tf.LookupException, tf.ConnectivityException):
            rospy.logerr("Failed to find camera to map transform")

        #TODO Use tranform and rotation to calculate 2D position of light in image

        x = 0
        y = 0

        return (x, y)

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """

        if(not self.has_image):
            self.prev_light_loc = None
            return False

        self.camera_image.encoding = "rgb8"

        # cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")
        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "passthrough")

        ##not quite sure why we would need the following line written by Udacity
        # x, y = self.project_to_image_plane(light.pose.pose.position)

        #TODO use light location to zoom in on traffic light in image
        #I believe we don't need this TODO

        #Get classification
        return self.light_classifier.get_classification(cv_image)

    def distance(self, a, b):
        return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)

    def get_first_traffic_light(self, lightsPose):
        nearest_light = None
        nearest_light_index = 1000000
        for light in lightsPose:
            light_indx = self.get_closest_waypoint(light)
            if light_indx < nearest_light_index:
                nearest_light = light
                nearest_light_index = light_indx
        return nearest_light, nearest_light_index

    def find_next_light(self, pose):
        nearest_light = None
        nearest_light_index = 1000000
        pose_indx = self.get_closest_waypoint(pose)
        print("car at ", pose_indx)
        for light in self.stop_lights_pos:
            light_indx = self.lights_cache[light]
            if pose_indx < light_indx and light_indx < nearest_light_index:
                nearest_light = light
                nearest_light_index = light_indx

        if not nearest_light:
            nearest_light = self.firstLight
            nearest_light_index = self.firstLightIndex
        return nearest_light, nearest_light_index

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        light = None
        light_wp = None

        # List of positions that correspond to the line to stop in front of for a given intersection
        # stop_line_positions = self.config['stop_line_positions']
        if self.firstLight is None:
            self.populate_cache()
            
        if(self.pose):
            light,light_wp = self.find_next_light(self.pose.pose)

        

        # todo we may want to check if the car is close enough to the light and only then get light state. to
        # do this uncomment the following line and comment out the one after it
        if light:
            dist = self.distance(self.pose.pose.position, light.position)
            if dist < 150:
                print("next traffic light at, ", light_wp, " classifying")
                state = self.get_light_state(light)
                print("light color ", self.get_color(state))
                return light_wp, state

        # self.waypoints = None
        return -1, TrafficLight.UNKNOWN

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
