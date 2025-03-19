#!/usr/bin/env python3

# potentially useful for part 1 of exercise 4

# import required libraries
import rospy
import os
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage, Image, CameraInfo
import numpy as np

import cv2 as cv
from cv_bridge import CvBridge

class ApriltagNode(DTROS):

    def __init__(self, node_name):
        super(ApriltagNode, self).__init__(node_name=node_name, node_type=NodeType.CONTROL)

        # add your code here

        # call navigation control node
        # static parameters
        self._vehicle_name = os.environ['VEHICLE_NAME']
        self._camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"
        self._camera_info = f"/{self._vehicle_name}/camera_node/camera_info"

        # initialize dt_apriltag detector
        # https://github.com/duckietown/lib-dt-apriltags

        # subscribe to camera feed
        self.K = None
        self.D = None
        self.sub_info = rospy.Subscriber(self._camera_info, CameraInfo, self.callback_info)
        self._bridge = CvBridge()
        self.disorted_image = None
        self.color_detect_image = None
        self.black_detect_image = None
        self.sub_image = rospy.Subscriber(self._camera_topic, CompressedImage, self.callback_image)

        # define other variables as needed
        # lane detection publishers
        self._custom_topic = f"/{self._vehicle_name}/custom_node/image/compressed"
        self.pub = rospy.Publisher(self._custom_topic, Image) # queue_size=10

        self._custom_topic_lane = f"/{self._vehicle_name}/custom_node/image/black"
        self.pub_lane = rospy.Publisher(self._custom_topic_lane, Image) # queue_size=10
    
    def callback_info(self, msg):
        rate = rospy.Rate(1)
        # https://stackoverflow.com/questions/55781120/subscribe-ros-image-and-camerainfo-sensor-msgs-format
        # http://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/CameraInfo.html
        # https://github.com/IntelRealSense/realsense-ros/issues/709ss
        self.K = np.array(msg.K).reshape(3, 3)
        self.D = np.array(msg.D)
        rospy.loginfo("Camera parameters received.")
        rate.sleep()

    def callback_image(self, msg):
        # add your code here
        
        # convert compressed image to CV2
        rate = rospy.Rate(3)
        if self.K is None:
            return
        image = self._bridge.compressed_imgmsg_to_cv2(msg)
        # undistort image
        dst = self.undistort_image(image)
        # preprocess image
        imageFrame = self.preprocess_image(dst).astype(np.uint8)
        self.disorted_image = imageFrame
        rospy.loginfo("Image Calibrated")

    def undistort_image(self, image):
        # convert JPEG bytes to CV image
        # rate = rospy.Rate(3)
        if self.K is None:
            return
        # https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
        h,w = image.shape[:2]
        newcameramtx, roi = cv.getOptimalNewCameraMatrix(self.K, self.D, (w,h), 1, (w,h))
        dst = cv.undistort(image, self.K, self.D, None, newcameramtx)
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]
        # rospy.loginfo("Image Calibrated")
        return dst
        # rate.sleep()
    
    def preprocess_image(self, raw_image):
        new_width = 400
        new_height = 300
        resized_image = cv.resize(raw_image, (new_width, new_height), interpolation = cv.INTER_AREA)
        blurred_image = cv.blur(resized_image, (5, 5)) 
        return blurred_image

    def sign_to_led(self, **kwargs):
        pass

    def process_image(self, **kwargs):
        pass

    def publish_augmented_img(self, **kwargs):
        pass

    def publish_leds(self, **kwargs):
        pass

    def detect_tag(self, **kwargs):
        pass


if __name__ == '__main__':
    # create the node
    node = ApriltagNode(node_name='apriltag_detector_node')
    rospy.spin()
    
