#!/usr/bin/env python3

# potentially useful for part 1 of exercise 4

# import required libraries
import rospy
import os
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage, Image, CameraInfo
import numpy as np
from duckietown_msgs.msg import LEDPattern 

import cv2 as cv
from cv_bridge import CvBridge
import dt_apriltags as aptag
from std_msgs.msg import Header, ColorRGBA 

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

        self.led_topic = f"/{self._vehicle_name}/led_emitter_node/led_pattern"
        self.led_pub = rospy.Publisher(self.led_topic, LEDPattern, queue_size=1)

        # define other variables as needed
        # lane detection publishers
        self._custom_topic = f"/{self._vehicle_name}/custom_node/image/compressed"
        self.pub = rospy.Publisher(self._custom_topic, Image) # queue_size=10

        self._custom_topic_augmented_image = f"/{self._vehicle_name}/custom_node/image/black"
        self.pub_augmented_image = rospy.Publisher(self._custom_topic_augmented_image , Image) # queue_size=10

        self.gray = None
        self.publish_augmented_img()

        
    
    def callback_info(self, msg):
        rate = rospy.Rate(1)
        # https://stackoverflow.com/questions/55781120/subscribe-ros-image-and-camerainfo-sensor-msgs-format
        # http://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/CameraInfo.html
        # https://github.com/IntelRealSense/realsense-ros/issues/709ss
        self.K = np.array(msg.K).reshape(3, 3)
        self.D = np.array(msg.D)
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
        self.process_image()
        self.detect_tag()


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

    def sign_to_led(self, tag_id):

        x = ()
        if int(tag_id) == 51:
            x = (0.0, 0.0, 1.0, 1.0)
        elif int(tag_id) == 162:
            x = (1.0, 0.0, 0.0, 1.0)
        elif int(tag_id) == 201:
            x = (0.0, 1.0, 0.0, 1.0)
        else: 
            x = (1.0, 1.0, 1.0, 1.0)
        
        self.publish_leds(x)
        return x

    def process_image(self):
        self.gray = cv.cvtColor(self.disorted_image, cv.COLOR_BGR2GRAY)
        pass

    def publish_augmented_img(self):
        rate = rospy.Rate(3)
        while not rospy.is_shutdown():       
            if self.gray is not None:
                image_msg = self._bridge.cv2_to_imgmsg(self.gray, encoding="8UC1")
                self.pub_augmented_image.publish(image_msg)
                
           
        rate.sleep()
        pass

    def publish_leds(self, x):      
        if self.gray is not None:
            msg = LEDPattern()
            msg.header = Header()
            msg.header.stamp = rospy.Time.now()
            color_msg = ColorRGBA()
            color_msg.r, color_msg.g, color_msg.b, color_msg.a = x


            # Set LED colors
            msg.rgb_vals = [color_msg] * 5
            self.led_pub.publish(msg) 
        pass

    def detect_tag(self):
        detector = aptag.Detector(families="tag36h11")
        results = detector.detect(self.gray)
           
        def area(r):
            # Use corners to compute polygon area
            (ptA, ptB, ptC, ptD) = r.corners
            return 0.5 * abs(
                ptA[0]*ptB[1] + ptB[0]*ptC[1] + ptC[0]*ptD[1] + ptD[0]*ptA[1]
                - ptB[0]*ptA[1] - ptC[0]*ptB[1] - ptD[0]*ptC[1] - ptA[0]*ptD[1]
            )

        largest_tag = max(results, key=area)

        # Extract corners
        (ptA, ptB, ptC, ptD) = largest_tag.corners
        ptA = (int(ptA[0]), int(ptA[1]))
        ptB = (int(ptB[0]), int(ptB[1]))
        ptC = (int(ptC[0]), int(ptC[1]))
        ptD = (int(ptD[0]), int(ptD[1]))

        # Draw bounding box
        cv.line(self.gray, ptA, ptB, (0, 255, 0), 2)
        cv.line(self.gray, ptB, ptC, (0, 255, 0), 2)
        cv.line(self.gray, ptC, ptD, (0, 255, 0), 2)
        cv.line(self.gray, ptD, ptA, (0, 255, 0), 2)

        # Draw tag ID at center
        (cX, cY) = (int(largest_tag.center[0]), int(largest_tag.center[1]))
        tag_id = str(largest_tag.tag_id)
        cv.putText(self.gray, tag_id, (cX - 10, cY + 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        self.sign_to_led(tag_id)
        

        return 


if __name__ == '__main__':
    # create the node
    node = ApriltagNode(node_name='apriltag_detector_node')
    rospy.spin()
    
