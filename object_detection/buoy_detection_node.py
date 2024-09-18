#!usr/bin/python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from realsense2_camera_msgs.msg import RGBD
from sailbot_msgs.msg import ObjectDetectionResults

from cv_bridge import CvBridge
from ultralytics import YOLO



TRAINED_IMAGE_SIZE = (640, 640)     # pixel width and height of the images that the model was trained on
IMAGE_CONFIDENCE = 0.5

class BuoyDetectionNode(Node):
    
    def __init__(self):
        super().__init__("buoy_detection")
        self.model = YOLO("weights/yolov9c4.pt")
        self.cv_bridge  = CvBridge()
        self.current_image_cv = None
        
        sensor_qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        self.rgbd_camera_listener = self.create_subscription(msg_type=RGBD, topic="/camera/camera/aligned_depth_to_color/image_raw", callback=self.rgbd_camera_image_callback, qos_profile=sensor_qos_profile)
        self.object_detection_results_publisher = self.create_publisher(msg_type=ObjectDetectionResults, topic="/object_detection_results", qos_profile=sensor_qos_profile)
        
        
        self.create_timer(timer_period_sec=0.1, callback=self.perform_inference)
        
        
    def rgbd_camera_image_callback(self, rgbd_image: RGBD):
        depth_image = rgbd_image.depth
        rgb_image = rgbd_image.rgb
        
        depth_image_cv = self.cv_bridge.imgmsg_to_cv2(depth_image, "rgb8")
        rgb_image_cv = self.cv_bridge.imgmsg_to_cv2(rgb_image, "rgb8")
        
        assert depth_image_cv.shape == rgb_image_cv.shape
        
        print(f"depth image shape: {depth_image_cv.shape}")
        
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
        
        depth_image_cv.resize(TRAINED_IMAGE_SIZE)
        rgb_image_cv.resize(TRAINED_IMAGE_SIZE)
        
        print(f"cropped image shape: {depth_image_cv.shape}")
        
        # depth_image_cv.show()
        
        
        
        
    def perform_inference(self):
        # https://docs.ultralytics.com/modes/predict/#inference-sources
        results = self.model.predict([self.current_image_cv,], conf=IMAGE_CONFIDENCE)  # return a list of Results objects

        # Process results list
        for result in results:
            boxes = result.boxes  # Boxes object for bounding box outputs
            print(f"boxes: {boxes}")
            masks = result.masks  # Masks object for segmentation masks outputs
            keypoints = result.keypoints  # Keypoints object for pose outputs
            probs = result.probs  # Probs object for classification outputs
            obb = result.obb  # Oriented boxes object for OBB outputs
            result.show()  # display to screen
            
        # TODO process these results properly
        # self.object_detection_results_publisher.publish()

        
        
def main():
    rclpy.init()
    buoy_detection_node = BuoyDetectionNode()
    rclpy.spin(buoy_detection_node)
    


if __name__ == "__main__": main()