import numpy as np
from ultralytics import YOLO
from fusion import *
from utils import *
from rosutils import *
import time
from visualization import *
from std_msgs.msg import Header, String
from sensor_msgs.msg import PointCloud2, Image, CompressedImage
import json
import threading
import copy

class YOLOv8Detector(ROSUtils):
    def __init__(self, model_path, tracking=False, PCA=False, calib_file="config/processing_config.yaml"):
        super().__init__()
        self.model = YOLO(model_path, task='segment')
        self.tracking = tracking
        self.pca = PCA
        self.last_ground_center_of_id = {}

        with open(calib_file, "r") as f:
            data = yaml.safe_load(f)

        self.erosion_factor = data["fusion"]["erosion_factor"]
        self.depth_factor = data["fusion"]["depth_factor"]

        self.imgpub = rospy.Publisher('/fused_image', CompressedImage, queue_size=1)
        self.depthpub = rospy.Publisher('/depths', String, queue_size=1)

        # Shared data for visualization thread
        self.latest_frame = None
        self.frame_lock = threading.Lock()

    def process_frame(self, frame, pts, lidar2camera, erosion_factor, depth_factor):
        if self.tracking:
            results = self.model.track(
                source=frame,
                verbose=False,
                show=False,
                persist=True,
                tracker='bytetrack.yaml',
                task='segment',
                conf=0.4,
                device='cuda:0'
            )
        else:
            results = self.model.predict(
                source=frame,
                verbose=False,
                show=False,
                device='cuda:0',
                task='segment',
                conf=0.4
            )

        r = results[0]
        print(f'got segmentation results: {time.time()}')
        boxes = r.boxes
        masks = r.masks

        points = pts.reshape((-1, 4))[:, 0:3]
        point_cloud = np.asarray(points)
        downsampled_points = point_cloud

        print(f'downsampled cloud: {time.time()}')
        pts_3D, pts_2D = filter_lidar_points(
            lidar2camera,
            downsampled_points,
            (frame.shape[1], frame.shape[0]),
            print_info=True
        )
        print(f'filtered lidar points, starting fusion: {time.time()}')

        all_corners_3D = []
        all_filtered_points_of_object = []
        all_object_IDs = []
        objects3d_data = []
        depth_dict = dict()
        objno = 0

        for j, cls in enumerate(boxes.cls.tolist()):
            box_id = int(boxes.id.tolist()[j]) if boxes.id is not None else None
            all_object_IDs.append(box_id)

            if masks is None or masks.xy[j].size == 0:
                continue

            fusion_result = lidar_camera_fusion(
                pts_3D, pts_2D, frame, masks.xy[j], int(cls),
                lidar2camera,
                erosion_factor=erosion_factor,
                depth_factor=depth_factor,
                PCA=self.pca
            )

            if fusion_result is not None:
                filtered_points_of_object, corners_3D, yaw = fusion_result

                avg_depth = np.mean(
                    np.sqrt(filtered_points_of_object[:, 0]**2 + filtered_points_of_object[:, 1]**2)
                )
                depth_dict[objno] = avg_depth
                objno += 1

                all_corners_3D.append(corners_3D)
                all_filtered_points_of_object.append(filtered_points_of_object)

                ROS_type = int(np.int32(cls))
                bottom_indices = np.argsort(corners_3D[:, 2])[:4]
                ROS_ground_center = np.mean(corners_3D[bottom_indices], axis=0)
                ROS_dimensions = np.ptp(corners_3D, axis=0)
                ROS_points = corners_3D
                time_between_frames = 0.1

                if box_id in self.last_ground_center_of_id and not np.array_equal(
                    self.last_ground_center_of_id[box_id], ROS_ground_center
                ):
                    ROS_direction, ROS_velocity = compute_relative_object_velocity(
                        self.last_ground_center_of_id[box_id],
                        ROS_ground_center,
                        time_between_frames
                    )
                else:
                    ROS_direction = None
                    ROS_velocity = None

                self.last_ground_center_of_id[box_id] = ROS_ground_center

                if (
                    ROS_type is not None and
                    ROS_ground_center is not None and
                    ROS_direction is not None and
                    ROS_dimensions is not None and
                    ROS_velocity is not None and
                    ROS_points is not None
                ):
                    objects3d_data.append([
                        ROS_type,
                        ROS_ground_center,
                        ROS_direction,
                        ROS_dimensions,
                        ROS_velocity,
                        ROS_points
                    ])

        print(f'fusion completed: {time.time()}')
        return (
            objects3d_data,
            all_corners_3D,
            pts_3D,
            pts_2D,
            all_filtered_points_of_object,
            all_object_IDs,
            depth_dict
        )

    def detection_loop(self):
        """
        Only thread that runs YOLO + fusion.
        Produces the rendered frame and publishes depth.
        """
        lidar2cam = LiDAR2Camera("config/cam_config.yaml")

        while not rospy.is_shutdown():
            print(f'starting cycle: getting data: {time.time()}')
            pts = self.getLidarPoints()
            frame = self.getCameraImage()

            if frame is None or pts is None:
                continue

            print(f'got lidar points + frame, starting frame processing: {time.time()}')

            (
                _,
                all_corners_3D,
                pts_3D,
                pts_2D,
                all_filtered_points_of_object,
                _,
                depth_dict
            ) = self.process_frame(frame, pts, lidar2cam, self.erosion_factor, self.depth_factor)

            #rendered_frame = frame.copy()
            rendered_frame = frame

            if len(all_corners_3D) > 0:
                for pred_corner_3D in all_corners_3D:
                    plot_projected_pred_bounding_boxes(
                        lidar2cam,
                        rendered_frame,
                        pred_corner_3D,
                        (0, 0, 255)
                    )

            if len(all_filtered_points_of_object) > 0:
                draw_projected_3D_points(
                    lidar2cam,
                    rendered_frame,
                    pts_3D,
                    pts_2D,
                    np.vstack(all_filtered_points_of_object)
                )
            # Publish depth here since detection thread already computed it
            if len(depth_dict) > 0:
                self.publish_depth(depth_dict)

            # Store latest rendered frame for publisher thread
            #with self.frame_lock:
            self.latest_frame = rendered_frame

            # Publish depth here since detection thread already computed it
            #if len(depth_dict) > 0:
            #    self.publish_depth(depth_dict)

    def visualization_loop(self, rate_hz=10):
        """
        Only publishes the latest already-rendered frame.
        No YOLO, no fusion, no sensor fetch.
        """
        rate = rospy.Rate(rate_hz)

        while not rospy.is_shutdown():
            frame_to_publish = None

            #with self.frame_lock:
            if self.latest_frame is not None:
                frame_to_publish = self.latest_frame.copy()

            if frame_to_publish is not None:
                self.publish_frame(frame_to_publish)

            rate.sleep()

    def publish_frame(self, frame):
        msg = CompressedImage()
        msg.header = Header()
        msg.header.stamp = rospy.Time.now()
        msg.format = 'jpeg'
        msg.data = np.array(cv2.imencode('.jpg', frame)[1]).tobytes()
        self.imgpub.publish(msg)

    def publish_depth(self, depth_dict):
        depthmsg = String()
        depthmsg.data = json.dumps(depth_dict)
        print(f'dictionary: {depthmsg.data}')
        self.depthpub.publish(depthmsg)
        print(f'depth published: {time.time()}')


if __name__ == "__main__":
    model_path = "../../visualnav-transformer/train/best_built-in-cam.engine"
    detector = YOLOv8Detector(model_path)

    time.sleep(3)

    detection_thread = threading.Thread(target=detector.detection_loop)
    visualization_thread = threading.Thread(target=detector.visualization_loop)

    detection_thread.start()
    visualization_thread.start()

    detection_thread.join()
    visualization_thread.join()

    print("end")
