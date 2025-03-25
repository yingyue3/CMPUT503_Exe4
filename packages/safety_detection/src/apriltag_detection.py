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
from std_msgs.msg import Header, ColorRGBA, Int32, String

class ApriltagNode(DTROS):

    def __init__(self, node_name):
        super(ApriltagNode, self).__init__(node_name=node_name, node_type=NodeType.CONTROL)
        self._vehicle_name = os.environ['VEHICLE_NAME']

        self.gray_topic = f"/{self._vehicle_name}/custom_node/image/gray"
        self.lane_sub = rospy.Subscriber(self.gray_topic, Image, self.callback_image)

        self.led_topic = f"/{self._vehicle_name}/led_emitter_node/led_pattern"
        self.led_pub = rospy.Publisher(self.led_topic, LEDPattern, queue_size=1)

        
        self.gray = None
        self._bridge = CvBridge()

        
    
   
    def callback_image(self, image):
        # add your code here
        
        self.gray = self._bridge.imgmsg_to_cv2(image) 

        self.detect_tag()
    
   
    
        
            
    def sign_to_led(self, tag_id):

        x = ()
        if int(tag_id) == 50 or int(tag_id) == 133:
            x = (0.0, 0.0, 1.0, 1.0)
        elif int(tag_id) == 22 or int(tag_id) == 21:
            x = (1.0, 0.0, 0.0, 1.0)
        elif int(tag_id) == 93 or int(tag_id) == 94:
            x = (0.0, 1.0, 0.0, 1.0)
        else: 
            x = (1.0, 1.0, 1.0, 1.0)
        
        self.publish_leds(x)
        return x

    def publish_augmented_img(self):   
        if self.gray is not None or self.black_detect_image is not None:
            # image_msg = self._bridge.cv2_to_imgmsg(self.gray, encoding="8UC1")
            # self.pub_augmented_image.publish(image_msg)

            black_msg = self._bridge.cv2_to_imgmsg(self.black_detect_image, encoding="8UC1")
            self.pub_black.publish(black_msg)
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
        # rospy.loginfo(results)
        if len(results) == 0:
            self.sign_to_led(1000)
            # rospy.loginfo("None")
            return
           
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
        # cv.line(self.gray, ptA, ptB, (0, 255, 0), 2)
        # cv.line(self.gray, ptB, ptC, (0, 255, 0), 2)
        # cv.line(self.gray, ptC, ptD, (0, 255, 0), 2)
        # cv.line(self.gray, ptD, ptA, (0, 255, 0), 2)

        # Draw tag ID at center
        (cX, cY) = (int(largest_tag.center[0]), int(largest_tag.center[1]))
        tag_id = str(largest_tag.tag_id)
        # cv.putText(self.gray, tag_id, (cX - 10, cY + 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        # rospy.loginfo(tag_id)

        self.sign_to_led(tag_id)
        # self.aprilid = int(tag_id)
        

        return 


if __name__ == '__main__':
    # create the node
    
    node = ApriltagNode(node_name='apriltag_detector_node')
    rospy.Rate(20)

    rospy.spin()
    
