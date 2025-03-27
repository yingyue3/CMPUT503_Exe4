#!/usr/bin/env python3

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

class LaneDetectionNode(DTROS):

    def __init__(self, node_name):
        super(LaneDetectionNode, self).__init__(node_name=node_name, node_type=NodeType.CONTROL)

        # add your code here
        # while bound
        self.white_lower = np.array([0, 0, 180], np.uint8) 
        self.white_upper = np.array([180, 40, 255], np.uint8)

        # yellow bound
        self.yellow_lower = np.array([20, 80, 100], np.uint8) 
        self.yellow_upper = np.array([40, 255, 255], np.uint8) 

        # color detection parameters in HSV format
        # Set range for red color
        self.red_lower = np.array([135, 80, 100], np.uint8) 
        self.red_upper = np.array([190, 255, 255], np.uint8)


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
        self.imageFrame = None
        self.sub_image = rospy.Subscriber(self._camera_topic, CompressedImage, self.callback_image)


        self._custom_topic_augmented_image = f"/{self._vehicle_name}/custom_node/image/gray"
        self.pub_augmented_image = rospy.Publisher(self._custom_topic_augmented_image , Image, queue_size=1)

        self._custom_topic_black = f"/{self._vehicle_name}/custom_node/image/black"
        self.pub_black = rospy.Publisher(self._custom_topic_black , Image, queue_size=1)

        self.red_lane_message = "No"
        self._custom_topic_red_lane = f"/{self._vehicle_name}/control_node/red_lane"
        self.pub_red_lane = rospy.Publisher(self._custom_topic_red_lane, String, queue_size = 1)

        self._custom_tag= f"/{self._vehicle_name}/control_node/tag"
        self.pub_tag = rospy.Publisher(self._custom_tag, String, queue_size = 1)
        self.x = (1.0, 1.0, 1.0, 1.0)
        self.prev_x = (1.0, 1.0, 1.0, 1.0)

        self.tag_id =1000

        self.tag_msg = self.tag_id
        self.image = None
        self.gray = None

        

    def detect_red_lane(self, imageFrame):
        hsvFrame = cv.cvtColor(imageFrame, cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsvFrame, self.red_lower, self.red_upper) 

        kernel = np.ones((5, 5), "uint8") 
        mask = cv.dilate(mask, kernel) 
        res = cv.bitwise_and(imageFrame, imageFrame, 
                                mask = mask) 
        contours, hierarchy = cv.findContours(mask, 
                                            cv.RETR_TREE, 
                                            cv.CHAIN_APPROX_SIMPLE)
        red_lane = False
        for pic, contour in enumerate(contours): 
            area = cv.contourArea(contour) 
            if(area > 110): 
                x, y, w, h = cv.boundingRect(contour) 
                red_lane = True
        if red_lane:
            self.red_lane_message = "Yes"
        else:
            self.red_lane_message = "No"
        return imageFrame

       
    def callback_info(self, msg):
        # https://stackoverflow.com/questions/55781120/subscribe-ros-image-and-camerainfo-sensor-msgs-format
        # http://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/CameraInfo.html
        # https://github.com/IntelRealSense/realsense-ros/issues/709ss
        self.K = np.array(msg.K).reshape(3, 3)
        self.D = np.array(msg.D)
        

    def callback_image(self, msg):
        # add your code here
        
        if self.K is None:
            return
        self.image = self._bridge.compressed_imgmsg_to_cv2(msg)
        # undistort image
        dst = self.undistort_image(self.image)
        # preprocess image
        self.disorted_image = dst
        self.imageFrame = self.preprocess_image(dst).astype(np.uint8)

    def start(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if self.imageFrame is None or self.disorted_image is None:
                continue


            self.color_detect_image = self.detect_red_lane(self.imageFrame)

            # April Tag detection code
            
            
            # PID control stuff
           
            self.black_detect_image = self.detect_lane(cv.blur(self.disorted_image, (5, 5)))

            self.process_image()
            # black_msg = self._bridge.cv2_to_imgmsg(self.black_detect_image, encoding="8UC1")
            # self.pub_augmented_image.publish(black_msg)
            # image_msg = self._bridge.cv2_to_imgmsg(self.gray, encoding="8UC1")
            self.publish_augmented_img()
            # end = rospy.Time.now()
            # rospy.loginfo("Publish time: %f sec", (end - start).to_sec())

    def undistort_image(self, image):
        # convert JPEG bytes to CV image
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
        
        resized_image = raw_image[339:339+100, 302:302+100]
        blurred_image = cv.blur(resized_image, (5, 5)) 
        return resized_image
    

    def detect_lane(self, imageFrame):
        # add your code here
        # potentially useful in question 2.1

        height = imageFrame.shape[0]
        imageFrame = imageFrame[height//2:-height//5, :, :]


        imageFrame = cv.GaussianBlur(imageFrame, (5, 5), 0)

        kernel = np.ones((5, 5), "uint8") 

        hsvFrame = cv.cvtColor(imageFrame, cv.COLOR_BGR2HSV)

        white_mask = cv.inRange(hsvFrame, self.white_lower, self.white_upper) 
        yellow_mask = cv.inRange(hsvFrame, self.yellow_lower, self.yellow_upper) 

        # For white color 
        white_mask = cv.dilate(white_mask, kernel) 
        res_white = cv.bitwise_and(imageFrame, imageFrame, 
                                mask = white_mask) 
        
        # For yellow color 
        yellow_mask = cv.dilate(yellow_mask, kernel) 
        res_yellow = cv.bitwise_and(imageFrame, imageFrame, 
                                mask = yellow_mask) 


        lane_mask = np.zeros_like(white_mask)  

        # Set yellow pixels to gray (128)
        # lane_mask[yellow_mask > 0] = 128  

        # Set white pixels to white (255)
        lane_mask[white_mask > 0] = 255 

        return lane_mask

   

    def process_image(self):
        if self.image is not None:
            self.gray = cv.cvtColor(self.image, cv.COLOR_BGR2GRAY)
            h, w = self.gray.shape
            self.gray = self.gray[h//4: -h//4, w//2:]
            self.detect_tag()
            self.tag_msg = self.tag_id
            self.pub_tag.publish(str(self.tag_msg))
            # rospy.loginfo(self.tag_id)
        pass

    def publish_augmented_img(self):   
        if self.gray is not None and self.black_detect_image is not None:
            image_msg = self._bridge.cv2_to_imgmsg(self.gray, encoding="8UC1")
            self.pub_augmented_image.publish(image_msg)

            black_msg = self._bridge.cv2_to_imgmsg(self.black_detect_image, encoding="8UC1")
            self.pub_black.publish(black_msg)

            self.pub_red_lane.publish(self.red_lane_message)

            # if self.red_lane_message == "Yes":
            #     for i in range(10):
            #         self.pub_red_lane.publish("No")
        pass

    
    
    def area(self, r):
            # Use corners to compute polygon area
            (ptA, ptB, ptC, ptD) = r.corners
            return 0.5 * abs(
                ptA[0]*ptB[1] + ptB[0]*ptC[1] + ptC[0]*ptD[1] + ptD[0]*ptA[1]
                - ptB[0]*ptA[1] - ptC[0]*ptB[1] - ptD[0]*ptC[1] - ptA[0]*ptD[1]
            )

    def detect_tag(self):
        if self.gray is not None:
            detector = aptag.Detector(families="tag36h11")
            results = detector.detect(self.gray)
            
            if len(results) == 0:
                self.tag_id = 1000
                return
            
            

            if len(results) > 1:
                largest_tag = max(results, key=self.area)
            else:
                largest_tag = results[0]

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

            # # Draw tag ID at center
            (cX, cY) = (int(largest_tag.center[0]), int(largest_tag.center[1]))
            self.tag_id = str(largest_tag.tag_id)
            cv.putText(self.gray, self.tag_id, (cX - 10, cY + 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            # rospy.loginfo(tag_id)
        
        return 


if __name__ == '__main__':
    # create the node
    node = LaneDetectionNode(node_name='lane_deection')
    node.start()

    rospy.spin()
    