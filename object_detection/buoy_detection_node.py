#!usr/bin/python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

# from realsense2_camera_msgs.msg import RGBD
from sensor_msgs.msg import Image
from sailbot_msgs.msg import ObjectDetectionResults

from cv_bridge import CvBridge
import torch
from ultralytics import YOLO
import cv2
import numpy as np


TRAINED_IMAGE_SIZE = (640, 640)     # pixel width and height of the images that the model was trained on
IMAGE_CONFIDENCE = 0.3

SHOULD_SAVE_IMAGES = True

class BuoyDetectionNode(Node):
    
    def __init__(self):
        super().__init__("buoy_detection")
        self.model = YOLO("/home/sailbot/sailbot_vt/src/object_detection/object_detection/weights/last3.pt")
        self.cv_bridge  = CvBridge()

        self.current_image_rgb = None
        self.current_image_depth = None

        self.image_to_save_index = 0 # images are saved in the format name[index].jpg so this just keeps track of the current index of the image so that we don't overwrite other images
        
        sensor_qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        self.depth_image_listener = self.create_subscription(msg_type=Image, topic="/camera/camera/aligned_depth_to_color/image_raw", callback=self.depth_image_callback, qos_profile=sensor_qos_profile)
        self.rgb_image_listener = self.create_subscription(msg_type=Image, topic="/camera/camera/color/image_raw", callback=self.rgb_image_callback, qos_profile=sensor_qos_profile)
        # self.object_detection_results_publisher = self.create_publisher(msg_type=ObjectDetectionResults, topic="/object_detection_results", qos_profile=sensor_qos_profile)
        
        
        self.create_timer(timer_period_sec=0.001, callback=self.perform_inference)
        
        
    def depth_image_callback(self, depth_image: Image):
        self.get_logger().info("got here depth")
        
        self.current_image_depth = self.cv_bridge.imgmsg_to_cv2(depth_image, "16UC1")
        
        # assert depth_image_cv.shape == rgb_image_cv.shape
        
        # self.get_logger().info(f"depth image shape: {rgbd_image.shape}")
        
        # smaller_size = min(depth_image_cv.shape)
        # larger_size = max(depth_image_cv.shape)

        # left = (larger_size-smaller_size)/2
        # right = left + smaller_size
        # top = 0
        # bottom = smaller_size
        
        # # TODO downscale the image so that the smallest dimension is 640p
        # # TODO: crop the image properly
        # cropped_depth_image_cv = depth_image_cv[left:right, top:bottom]
        # cropped_rgb_image_cv = rgb_image_cv
        
        # depth_image_cv.resize(TRAINED_IMAGE_SIZE)
        # rgb_image_cv.resize(TRAINED_IMAGE_SIZE)
        
        # print(f"cropped image shape: {depth_image_cv.shape}")
        # cv2.imwrite('rgb_image.jpg', rgb_image_cv) 
        # cv2.imwrite('depth_image.jpg', depth_image_cv) 

        # depth_image_cv.show()
        
        
        
    def rgb_image_callback(self, rgb_image: Image):
        self.get_logger().info("got here rgb")

        self.current_image_rgb = self.cv_bridge.imgmsg_to_cv2(rgb_image, "rgb8")
        
        self.current_image_rgb = self.current_image_rgb[80:1200,40:680] # crop the image to 640,640
        
        # swap red and blue channels for correction
        red = self.current_image_rgb[:,:,2].copy()
        blue = self.current_image_rgb[:,:,0].copy()
        self.current_image_rgb[:,:,0] = red
        self.current_image_rgb[:,:,2] = blue



    def perform_inference(self):
        # https://docs.ultralytics.com/modes/predict/#inference-sources
        if self.current_image_rgb is None: return
        results = self.model.predict([self.current_image_rgb,], conf=IMAGE_CONFIDENCE)  # return a list of Results objects



        # Process results list
        for result in results:
            boxes = result.boxes  # Boxes object for bounding box outputs
            print(f"boxes: {boxes}")
            masks = result.masks  # Masks object for segmentation masks outputs
            keypoints = result.keypoints  # Keypoints object for pose outputs
            probs = result.probs  # Probs object for classification outputs
            obb = result.obb  # Oriented boxes object for OBB outputs

            if SHOULD_SAVE_IMAGES:
                result.save(f"cv_results/result_{self.image_to_save_index}.png")  # display to screen
                self.image_to_save_index += 1
            
        # TODO process these results properly
        # self.object_detection_results_publisher.publish()

        
        
def main():
    rclpy.init()
    buoy_detection_node = BuoyDetectionNode()
    rclpy.spin(buoy_detection_node)
    


if __name__ == "__main__": main()