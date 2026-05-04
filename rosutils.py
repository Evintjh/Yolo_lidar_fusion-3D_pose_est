import rospy
import numpy as np
from sensor_msgs.msg import PointCloud2, Image, CompressedImage
import sensor_msgs.point_cloud2 as pc2
import cv2
import time
# from cv_bridge import CvBridge, CvBridgeError

class ROSUtils:
    def __init__(self):
        rospy.init_node('listener', anonymous=True)
        self.arraydata = np.array([])
        self.imagedata = None
        rospy.Subscriber("/lidar_points", PointCloud2, self.ros_callback, queue_size=1)
        rospy.Subscriber("/camera/image/compressed", CompressedImage, self.ros_imagecallback, queue_size=1)
        rospy.spin()

    def ros_callback(self, data):
        point_generator = pc2.read_points(data, field_names = ("x", "y", "z", "intensity"), skip_nans=True)
        points_list = list(point_generator)
        self.arraydata = np.array(points_list)
        
    def ros_imagecallback(self, data: CompressedImage):
        try:
            np_arr = np.frombuffer(data.data, np.uint8)
            # print(np_arr)
            self.imagedata = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        except Exception as e:
            print(e)

    def getLidarPoints(self):
        return self.arraydata
    def getCameraImage(self):
        return self.imagedata

if __name__ == "__main__":
    try:
        ros = ROSUtils()
        i = 10
        while i > 0:
            time.sleep(1)
            output = ros.getLidarPoints()
            imgout = ros.getCameraImage()
            print(output.size)
            print(imgout)
            i = i - 1

    except Exception as e:
        print(f'exception: {e}')