# 3D Visual SLAM

Using accelormeter, gyroscope data and stereo camrea data collected by [KITTI](http://www.cvlibs.net/datasets/kitti/) to recover the trajectory of the vehicles at each time step and obtained its landmark map based on a `Extended Kalman Filter` model.<br>


There are total three implementation: <br>
1. Localization via EKF predict<br>
2. Mapping via EKF update<br>
3. visual SLAM vis EKF predict and update

![] (/3D_visual_SLAM/Results/mapping_27.png)

