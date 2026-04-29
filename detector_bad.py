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

class YOLOv8Detector(ROSUtils):
    def __init__(self, model_path, tracking=False, PCA=False):
        super().__init__()
        self.model = YOLO(model_path, task='segment')
        self.tracking = tracking
        self.pca = PCA
        self.last_ground_center_of_id = {}
        # rospy.init_node('image_publisher', anonymous=True)
        self.imgpub = rospy.Publisher('/fused_image', CompressedImage, queue_size=1)
        self.depthpub = rospy.Publisher('/depths', String, queue_size=1)
        # rospy.spin()

    def process_frame(self, frame, pts, lidar2camera, erosion_factor, depth_factor):
        if self.tracking:
            results = self.model.track(
                source=frame,
                classes=[0, 1, 2, 3, 5, 6, 7],
                verbose=False,
                show=False,
                persist=True,
                tracker='bytetrack.yaml'
            )
        else:
            results = self.model.predict(
                source=frame,
                classes=[0, 1, 2, 3, 5, 6, 7],
                verbose=False,
                show=False,
            )

        # Get the results from the YOLOv8-seg model
        r = results[0]
        boxes = r.boxes  # Boxes object for bbox outputs
        masks = r.masks  # Masks object for segment masks outputs

        # Preprocess LiDAR point cloud
        points = np.fromfile(pts, dtype=np.float32).reshape((-1, 4))[:, 0:3]
        point_cloud = np.asarray(points)
        pts_3D, pts_2D = filter_lidar_points(lidar2camera, point_cloud, (frame.shape[1], frame.shape[0]))

        # For each object detected by the YOLOv8 model, fuse and process it
        all_corners_3D = []
        all_filtered_points_of_object = []
        all_object_IDs = []
        objects3d_data = []
        for j, cls in enumerate(boxes.cls.tolist()):
            conf = boxes.conf.tolist()[j] if boxes.conf is not None else None
            box_id = int(boxes.id.tolist()[j]) if boxes.id is not None else None

            all_object_IDs.append(box_id)

            # Check if the mask is empty before processing
            if masks.xy[j].size == 0:
                continue

            # Pass the segmentation mask to the fusion function
            fusion_result = lidar_camera_fusion(pts_3D, pts_2D, frame, masks.xy[j], int(cls), lidar2camera, erosion_factor=erosion_factor, depth_factor=depth_factor, PCA=self.pca)

            # If the fusion is successfull, retrieve relevant bbox data (e.g. for RoboCar)
            if fusion_result is not None:
                filtered_points_of_object, corners_3D, yaw = fusion_result

                all_corners_3D.append(corners_3D)
                all_filtered_points_of_object.append(filtered_points_of_object)

                # Retrieve the ROS data (e.g. relevant for RoboCar)
                ROS_type = int(np.int32(cls))
                bottom_indices = np.argsort(corners_3D[:, 2])[:4]
                ROS_ground_center = np.mean(corners_3D[bottom_indices], axis=0)
                ROS_dimensions = np.ptp(corners_3D, axis=0)                
                ROS_points = corners_3D
                time_between_frames = 0.1

                # Compute the velocity and direction (only available with tracking)
                if box_id in self.last_ground_center_of_id and not np.array_equal(self.last_ground_center_of_id[box_id], ROS_ground_center):
                    ROS_direction, ROS_velocity = compute_relative_object_velocity(self.last_ground_center_of_id[box_id], ROS_ground_center, time_between_frames)
                else:
                    ROS_direction = None
                    ROS_velocity = None

                self.last_ground_center_of_id[box_id] = ROS_ground_center

                # Save the ROS information of the current object and append it to an array that contains all information of all objects in the frame
                if ROS_type is not None and ROS_ground_center is not None and ROS_direction is not None and ROS_dimensions is not None and ROS_velocity is not None and ROS_points is not None:
                    objects3d_data.append([ROS_type, ROS_ground_center, ROS_direction, ROS_dimensions, ROS_velocity, ROS_points])


        return objects3d_data, all_corners_3D, pts_3D, pts_2D, all_filtered_points_of_object, all_object_IDs
    
    def get_IoU_results(self, frame, pts, lidar2camera, erosion_factor, depth_factor):
        if self.tracking:
            results = self.model.track(
                source=frame,
                classes=[0, 1, 2, 3, 5, 6, 7],
                verbose=False,
                show=False,
                persist=True,
                tracker='bytetrack.yaml'
            )
        else:  
            results = self.model.predict(
                source=frame,
                classes=[0, 1, 2, 3, 5, 6, 7],
                verbose=False,
                show=False,
            )

        # Get the results from the YOLOv8-seg model
        r = results[0]
        boxes = r.boxes  # Boxes object for bbox outputs
        masks = r.masks  # Masks object for segment masks outputs

        # Preprocess LiDAR point cloud
        points = np.fromfile(pts, dtype=np.float32).reshape((-1, 4))[:, 0:3]
        point_cloud = np.asarray(points)
        pts_3D, pts_2D = filter_lidar_points(lidar2camera, point_cloud, (frame.shape[1], frame.shape[0]))

        # For each object detected by the YOLOv8 model, fuse and process it
        all_corners_3D = []
        all_filtered_points_of_object = []
        objects3d_data = []
        for j, cls in enumerate(boxes.cls.tolist()):
            conf = boxes.conf.tolist()[j] if boxes.conf is not None else None
            box_id = int(boxes.id.tolist()[j]) if boxes.id is not None else None

            # Check if the mask is empty before processing
            if masks.xy[j].size == 0:
                continue

            # Pass the segmentation mask to the fusion function
            fusion_result = lidar_camera_fusion(pts_3D, pts_2D, frame, masks.xy[j], int(cls), lidar2camera, erosion_factor=erosion_factor, depth_factor=depth_factor, PCA=self.pca)

            # If the fusion is successfull, retrieve the relevant data for the IoU computation with KITTI GT boxes
            if fusion_result is not None:
                filtered_points_of_object, corners_3D, yaw = fusion_result

                all_corners_3D.append(corners_3D)
                all_filtered_points_of_object.append(filtered_points_of_object)

                if cls == 0:
                    type = "Pedestrian"
                elif cls == 1:
                    type = "Cyclist"
                elif cls == 2:
                    type = "Car"
                else:
                    type = "DontCare"

                # Ground Center is the center of the bottom bbox side, thus of the 4 corners with the lowest z value (in LiDAR coordinates) 
                bottom_indices = np.argsort(corners_3D[:, 2])[:4]
                ground_center = np.mean(corners_3D[bottom_indices], axis=0)

                # Get the bbox dimensions in l, w, h format
                dimensions = np.ptp(corners_3D, axis=0)

                # Append relevant information to array that is later returned
                objects3d_data.append([type, ground_center, dimensions, yaw])

        return objects3d_data, all_corners_3D, pts_3D, pts_2D, all_filtered_points_of_object 

    def process_frame2(self, frame, pts, lidar2camera, erosion_factor, depth_factor):
        if self.tracking:
            results = self.model.track(
                source=frame,
                # classes=[0, 1, 2, 3, 4, 5, 6, 7],
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
                # classes=[0, 1, 2, 3, 4, 5, 6, 7],
                verbose=False,
                show=False,
                device='cuda:0',
                task='segment',
                conf=0.4
            )
        # Get the results from the YOLOv8-seg model
        r = results[0]
        print(f'got segmentation results: {time.time()}')
        boxes = r.boxes  # Boxes object for bbox outputs
        masks = r.masks  # Masks object for segment masks outputs
        # print(f'mask: {masks}')
        # Preprocess LiDAR point cloud
        # pts from lidar topic instead of static file
        points = pts.reshape((-1, 4))[:, 0:3]
        point_cloud = np.asarray(points)
        # Downsample point cloud
        #pcd = o3d.geometry.PointCloud()
        #pcd.points = o3d.utility.Vector3dVector(point_cloud)
        #voxel_size = 0.3
        #downsampled_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        #downsampled_points = np.asarray(downsampled_pcd.points)
        downsampled_points = point_cloud
        print(f'downsampled cloud: {time.time()}')
        pts_3D, pts_2D = filter_lidar_points(lidar2camera, downsampled_points, (frame.shape[1], frame.shape[0]), print_info=True)
        print(f'filtered lidar points, starting fusion: {time.time()}')
        all_corners_3D = []
        all_filtered_points_of_object = []
        all_object_IDs = []
        objects3d_data = []
        depth_dict = dict()
        objno = 0
        for j, cls in enumerate(boxes.cls.tolist()):
            conf = boxes.conf.tolist()[j] if boxes.conf is not None else None
            box_id = int(boxes.id.tolist()[j]) if boxes.id is not None else None

            all_object_IDs.append(box_id)

            # Check if the mask is empty before processing
            if masks.xy[j].size == 0:
                continue
            # Pass the segmentation mask to the fusion function
            fusion_result = lidar_camera_fusion(pts_3D, pts_2D, frame, masks.xy[j], int(cls), lidar2camera, erosion_factor=erosion_factor, depth_factor=depth_factor, PCA=self.pca)

            # If the fusion is successfull, retrieve relevant bbox data (e.g. for RoboCar)
            if fusion_result is not None:
                filtered_points_of_object, corners_3D, yaw = fusion_result
                
                ## filtered points average depth
                avg_depth = np.mean(np.sqrt(filtered_points_of_object[:, 0]**2 + filtered_points_of_object[:, 1]**2))
                avg_depth_x = np.mean(np.sqrt(filtered_points_of_object[:, 0]**2))
                print(f'avg_depth_x: {avg_depth_x}')
                print(f'avg_depth: {avg_depth}')
                depth_dict[objno] = avg_depth
                objno += 1
                all_corners_3D.append(corners_3D)
                all_filtered_points_of_object.append(filtered_points_of_object)

                # Retrieve the ROS data (e.g. relevant for RoboCar)
                ROS_type = int(np.int32(cls))
                bottom_indices = np.argsort(corners_3D[:, 2])[:4]
                ROS_ground_center = np.mean(corners_3D[bottom_indices], axis=0)
                ROS_dimensions = np.ptp(corners_3D, axis=0)                
                ROS_points = corners_3D
                time_between_frames = 0.1

                # Compute the velocity and direction (only available with tracking)
                if box_id in self.last_ground_center_of_id and not np.array_equal(self.last_ground_center_of_id[box_id], ROS_ground_center):
                    ROS_direction, ROS_velocity = compute_relative_object_velocity(self.last_ground_center_of_id[box_id], ROS_ground_center, time_between_frames)
                else:
                    ROS_direction = None
                    ROS_velocity = None

                self.last_ground_center_of_id[box_id] = ROS_ground_center

                # Save the ROS information of the current object and append it to an array that contains all information of all objects in the frame
                if ROS_type is not None and ROS_ground_center is not None and ROS_direction is not None and ROS_dimensions is not None and ROS_velocity is not None and ROS_points is not None:
                    objects3d_data.append([ROS_type, ROS_ground_center, ROS_direction, ROS_dimensions, ROS_velocity, ROS_points])
        print(f'fusion completed: {time.time()}')
        return objects3d_data, all_corners_3D, pts_3D, pts_2D, all_filtered_points_of_object, all_object_IDs, depth_dict

    def detectionthread(self, threadtype):
        while not rospy.is_shutdown():
            print(f'starting cycle: getting data: {time.time()}')
            pts = self.getLidarPoints()
            frame = self.getCameraImage()
            print(f'got lidar points + frame, starting frame processing: {time.time()}')
            lidar2cam = LiDAR2Camera("") # no need for calib path as the transformation values are hardcoded
            _, all_corners_3D, pts_3D, pts_2D, all_filtered_points_of_object, _, depth_dict = self.process_frame2(frame, pts, lidar2cam, 25, 20)
            print(f'processed frame, drawing bounding boxes: {time.time()}')
            # Draw the predicted bounding boxes onto the frame
            #for pred_corner_3D in all_corners_3D:
                #plot_projected_pred_bounding_boxes(lidar2cam, frame, pred_corner_3D, (0, 0, 255))
            print(f'predicted bounding boxes: {time.time()}')
            if (len(all_filtered_points_of_object) > 0):
                print(f'dictionary: {depth_dict}')
                # Draw the projected LiDAR points of the detected objects onto the frame
                #draw_projected_3D_points(lidar2cam, frame, pts_3D, pts_2D, np.vstack(all_filtered_points_of_object))
                print(f'draw projected 3D points: {time.time()}')
                # Create a bev representation
                # bev = create_BEV(all_filtered_points_of_object, all_corners_3D)
                print(f'create BEV: {time.time()}')

                # Save the combined image for visualization
                print(f'creating combined image: {time.time()}')
                # combined_image = create_combined_image_for_publishing(frame, bev)
                print(f'combined image obtained: {time.time()}')
                # result, encoded_image_buffer = cv2.imencode('.jpg', combined_image)
                # print(f'result: {result}')
                # if result:
                if (threadtype == "frame"):
                    # Draw the predicted bounding boxes onto the frame
                    for pred_corner_3D in all_corners_3D:
                        plot_projected_pred_bounding_boxes(lidar2cam, frame, pred_corner_3D, (0, 0, 255))
                    # Draw the projected LiDAR points of the detected objects onto the frame
                    draw_projected_3D_points(lidar2cam, frame, pts_3D, pts_2D, np.vstack(all_filtered_points_of_object))
                    print(f'draw projected 3D points: {time.time()}')
                    # Create a bev representation
                    # bev = create_BEV(all_filtered_points_of_object, all_corners_3D)
                    print(f'create BEV: {time.time()}')
                    self.publish_frame(frame)
                elif (threadtype == "depth"):
                    self.publish_depth(depth_dict)
                

    def publish_frame(self, frame):
        # Create a CompressedImage message
        msg = CompressedImage()
        msg.header = Header()
        msg.header.stamp = rospy.Time.now()
        msg.format = 'jpeg'
        # Convert the numpy array to a list of bytes
        msg.data = np.array(cv2.imencode('.jpg', frame)[1]).tobytes() 
        self.imgpub.publish(msg)

    def publish_depth(self, depth_dict):
        depthmsg = String()
        depthmsg.data = json.dumps(depth_dict)
        print(f'dictionary: {depthmsg.data}')
        self.depthpub.publish(depthmsg)
        print(f'image published: {time.time()}')

if __name__ == "__main__":
    model_path = "../../visualnav-transformer/train/best_built-in-cam.engine"
    detector = YOLOv8Detector(model_path)
    time.sleep(3)
    #detector.detectionthread()
    tf = threading.Thread(target=detector.detectionthread, args=("frame",))
    #td = threading.Thread(target=detector.detectionthread, args=("depth",))
    tf.start()
    #td.start()
    tf.join()
    #td.join()
    print("end")
