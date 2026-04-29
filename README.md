# YOLO-LiDAR Fusion

<img width="621" height="503" alt="Screenshot 2026-04-07 at 7 15 28 PM" src="https://github.com/user-attachments/assets/a495cd7c-a3a4-4d4a-ad9c-19f6f7441569" />
<img width="755" height="425" alt="Screenshot 2026-04-08 at 5 02 25 PM" src="https://github.com/user-attachments/assets/9f7a8170-f11d-4216-8864-8c7bb7e67620" />
<img width="755" height="425" alt="Screenshot 2026-04-08 at 5 04 21 PM" src="https://github.com/user-attachments/assets/c57d3dbe-c003-4a1f-9fcb-b99401914467" />


## 1. Overview
Original readme can be found on the [GitHub page](https://github.com/TimKie/YOLO-LiDAR-Fusion)
- This repo combines yoloE-seg model with livox lidar pcls

## 2. Usage
1. After connecting to X3002715, start the virtual environment

   ```shell
   cd yololidarmiscdev
   source env38/bin/activate
   ```

2. Run the detector.py code

   ```shell
   cd code
   python3 detector.py
   ```

3. The program subscribes to /lidar_points and /camera/image/compressed. Restart the camera stream in 192.168.1.105 for the latter.

Messages are published to /fused_image (frame data) and /depths (lidar point depth).

4. Parameters can be modified directly in the code 
   
   - **--erosion**: specifies the amount of erosion used by the model (smaller value --> higher erosion) (default: 25)
   
   Update the erosion argument value in the process_frame2() function call in detectionthread function of detector.py

   - **--depth**: specifies the depth filter factor used by the model (smaller value --> more aggressive filtering) (default: 20)

   Update the depth argument value in the process_frame2() function call in detectionthread function of detector.py
  
   - **--pca**: specifies whether PCA should be used to create the 3D bounding boxes for all detected objects (default: False)

   Update the PCA argument value in the YOLOv8Detector initialisation in \_\_main\_\_ function of detector.py

5. Other features can be enabled/disabled/modified in the code

   - Downsampling: (un)comment the lines above downsampled_points = point_cloud in process_frame2 function of detector.py

   - Draw LiDAR points: (un)comment the draw_projected_3D_points function in detectionthread function of detector.py

   - Draw ALL LiDAR points, including outside of masks: (un)comment the pts_to_draw_2D = lidar2cam.convert_3D_to_2D(np.array(**FOV_pts_3D**), print_info=False) line in draw_projected_3D_points function of visualization.py

   - Calibration matrices: these are hardcoded in the LiDAR2Camera initialisation function in calibration.py

   - Model: change the model in \_\_main\_\_ function of detector.py

All the best
