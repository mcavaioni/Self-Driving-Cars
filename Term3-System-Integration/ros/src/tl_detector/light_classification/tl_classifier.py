from styx_msgs.msg import TrafficLight

import numpy as np
import os
import sys
import tensorflow as tf
from collections import defaultdict, Counter
from io import StringIO
import time

label_name = [ 'RED', 'YELLOW', 'GREEN' ]
label_num_ref = [ 1, 2, 3 ]



# def most_common(lst):
#         return max(set(lst), key=lst.count)

class TLClassifier(object):
    def __init__(self, CKPT):

        #from: https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb        
        self.detection_graph = tf.Graph()

        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()

            with tf.gfile.GFile(CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        self.sess = tf.Session(graph=self.detection_graph)

        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')

        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')

        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

    

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """

        if self.sess is None: return TrafficLight.UNKNOWN      

        image_np_expanded = np.expand_dims(image, axis=0)

        time0 = time.time()

        # Actual detection.
        (boxes, scores, classes, num) = self.sess.run(
          [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
          feed_dict={self.image_tensor: image_np_expanded})

        time1 = time.time()

        # print("Classified image in ", (time1 - time0) * 1000, "ms")

        array_scores = np.squeeze(scores)
        classes = np.squeeze(classes).astype(np.int32)
        
        # above_thrshold_class = []
        # score_threshold = 0.5
        # for score_idx in range(len(array_scores)):
        #     if array_scores[score_idx] > score_threshold:
        #         above_thrshold_class.append(classes[score_idx])


        # self.current_light = TrafficLight.UNKNOWN

        # if most_common(above_thrshold_class) == 1:
        #     return TrafficLight.RED
        # elif most_common(above_thrshold_class) == 2:
        #     return TrafficLight.YELLOW
        # elif most_common(above_thrshold_class) == 3:
        #     return TrafficLight.GREEN

        score_threshold = 0.5
        self.current_light = TrafficLight.UNKNOWN
        if array_scores[0] > score_threshold:
            if classes[0] == 1:
                return TrafficLight.RED
            elif classes[0] == 2:
                return TrafficLight.YELLOW
            elif classes[0] == 3:
                return TrafficLight.GREEN


        return self.current_light


