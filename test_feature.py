## maintainer: Jingyu Song - jingyuso@umich.edu #####

import rospy
from cv_bridge import CvBridge, CvBridgeError
import message_filters
import cv2
from sensor_msgs.msg import Image
import numpy as np

class test_correspondence:
    def __init__(self):
        self.bridge = CvBridge()
        image_sub = message_filters.Subscriber("camera/color/image_raw", Image)
        depth_sub = message_filters.Subscriber("camera/aligned_depth_to_color/image_raw", Image)
        self.image_count = 0
        self.orb = cv2.ORB_create()
        # self.matcher = cv2.BFMatcher()
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)   # or pass empty dictionary

        self.matcher = cv2.FlannBasedMatcher(index_params,search_params)


        self.pub = rospy.Publisher("correspondence_image", Image, queue_size=2)

        rospy.init_node('test_correspondence', anonymous=True)
        self.ts = message_filters.ApproximateTimeSynchronizer([image_sub, depth_sub], 1, 0.5)
        self.ts.registerCallback(self.callback)
        rospy.spin()

    def callback(self, rgb_data, depth_data):
        # print("in the callback")
        # print(self.image_count)
        image = self.bridge.imgmsg_to_cv2(rgb_data, "bgr8")
        depth_image = self.bridge.imgmsg_to_cv2(depth_data, "32FC1")
        depth_array = np.array(depth_image, dtype=np.float32)
        cv2.normalize(depth_array, depth_array, 0, 1, cv2.NORM_MINMAX)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if self.image_count == 0:
            self.prev_image = gray
            self.image_count = self.image_count + 1
            return
        else:
            self.current_image = gray
        
        queryKeypoints, queryDescriptors = self.orb.detectAndCompute(self.current_image,None)
        trainKeypoints, trainDescriptors = self.orb.detectAndCompute(self.prev_image,None)

        
        matches = self.matcher.knnMatch(queryDescriptors,trainDescriptors, k=2)
        print(len(queryKeypoints))
        final_img = cv2.drawMatches(self.current_image, queryKeypoints,
            self.prev_image, trainKeypoints, matches[:100],None)
        
        self.pub.publish(self.bridge.cv2_to_imgmsg(final_img, 'bgr8'))
        self.prev_image = self.current_image
        self.image_count = self.image_count + 1
        
        
    
if __name__ == '__main__':
    test_correspondence()
    