# EC523 Project: Dense High-Quality 3D Reconstruction

## Overall pipeline
### 1.Depth Estimation
- For MVS-Net:
1.
2.
3.
- For Mono-Depth:
1.
2.
3.
- For ZoeDepth:
1. First git clone the original version of ZoeDepth
2. Replace the ZoeDepth_Loss_Modified.py file and ZoeDepth_Training_Json config file in original ZoeDepth folders
3. Run the training code with MidAir Dataset

### 2.Get Sparse 3D Point Cloud
- Install COLMAP to get sparse 3D point cloud for dataset
- Export predicted camera intrinsic parameters
- Using Agi2nerf.py to get json files containing corresponding  

### 3.Generate and Estimate Camera Intrinsic Parameters
- Using Agisoft Metashape to optimize sparse point could
- Export estimated camera intrinsic parameters
- make sure COLMAP and Agisoft get almost the same to do validation

### 4.NeRF-Studio
- Adding output of depth estimation path of all those different models into previous json file
which contains all spacial and camera parameters
- With the help of depth map as guidance, NeRF start to training on sparse 3D point cloud and do
volumn rendering. During the training process, the camera position is optimized to the best after
iteratons.

## Models

### Multi-view Stereo Network
<img src = "https://user-images.githubusercontent.com/81452190/236362485-00228576-65fc-4f7a-ae90-6858b54af813.png" width = "600px">

### Monocular Depth Estimation
<img src = "https://user-images.githubusercontent.com/81452190/236362529-737fb5da-62a2-4829-a87d-0bae24249463.png" width = "600px">

### Zoe-Depth
<img src = "https://user-images.githubusercontent.com/81452190/236362552-f4c86c71-eb4e-4b84-a955-c9b1cb951e29.png" width = "600px">


## Dataset

### MidAir
<img src = "https://user-images.githubusercontent.com/81452190/236362715-4320262d-fbd8-4e6c-8375-9e5b78482208.png" width = "400px">
<img src = "https://user-images.githubusercontent.com/81452190/236362747-0e178560-7d14-4328-8351-e83e5c924a32.png" width = "400px">

### B1
<img src = "https://user-images.githubusercontent.com/81452190/236362801-916762d0-cc3d-46c7-849e-33e9e2f4e287.png" width = "400px">

### B2
<img src = "https://user-images.githubusercontent.com/81452190/236362876-fdfce51a-679f-4441-a60a-b1bedf47819d.png" width = "400px">

## Result

## Reference
