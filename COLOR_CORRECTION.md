# Task 9.1 - Stereo Vision

Stereo depth estimation using OpenCV StereoBM.

## What it does
- Parses Middlebury-style calib.txt (fx, fy, cx, cy, baseline, doffs)
- Computes a disparity map from rectified left/right image pairs
- Estimates real-world depth at a chosen pixel
- Reconstructs a colored 3D point cloud and saves it as .ply

## Run
python stereo_vision.py
