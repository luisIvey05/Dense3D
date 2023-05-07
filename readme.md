# EC523 Project: Dense High-Quality 3D Reconstruction
<h4>
<a href="#intro"> Overall Pipeline </a> | 
<a href="#mod">Models</a> |
<a href="#data">Datasets</a> |
<a href="#met">Metrics</a> |
<a href="#res">Results</a> |
<a href = "#ref">References</a>
</h4>

<p id="intro"> </p>


## Overall Pipeline
### 1.Depth Estimation
- For MVS-Net:
Please, follow the instructions inside vis_mvsnet.ipynb.

- For Mono-Depth:
1. Git clone monodepth2 here: https://github.com/nianticlabs/monodepth2
2. Replace all files under ./monodepth
3. Run the training codes for midair dataset.
4. See more details in ./monodepth/README.md
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

<p id="mod"> </p>



## Models

### Multi-view Stereo Network
<img src = "https://user-images.githubusercontent.com/81452190/236362485-00228576-65fc-4f7a-ae90-6858b54af813.png" width = "600px">


Vis-MVSNet is a deep learning framework for dense multi-view stereo reconstruction. It has achieved state-of-the-art performance on several benchmark datasets. However, its performance can be further improved by fine-tuning on custom datasets. We fine-tuned Vis-MVSNet on our own dataset by modifying the configuration file and using Python scripts from the Vis-MVSNet GitHub repository to preprocess the images and create a list of neighboring images. This resulted in a significant improvement in performance compared to the original Vis-MVSNet and other existing methods.

### Monocular Depth Estimation
<img src = "https://user-images.githubusercontent.com/81452190/236362529-737fb5da-62a2-4829-a87d-0bae24249463.png" width = "600px">

Monodepth is a popular approach for depth estimation from a single RGB image. The first version of this self-supervised learning framework explored both photometric and geometric constraints to learn depth estimation from unlabeled video sequences. The architecture in the image consists of a depth network that predicts the depth map from an input image and a pose network that estimates the camera motion between frames. The task is expressed as the minimization of a photometric reprojection loss, which is a reconstruction error of the source view. Monodepth2 introduces three main contributions: a minimum reprojection loss, an auto-masking loss to ignore confusing stationary pixels, and a full-resolution multi-scale sampling method. Extensive experiments on KITTI 2015 stereo dataset demonstrate that Monodepth2 outperforms existing self-supervised methods and achieves state-of-the-art results on several depth estimation tasks.

### Zoe-Depth
<img src = "https://user-images.githubusercontent.com/81452190/236362552-f4c86c71-eb4e-4b84-a955-c9b1cb951e29.png" width = "600px">

ZoeDepth is a novel approach to depth estimation that addresses the limitations of absolute depth maps by leveraging relative depth cues. This method involves two stages: a self-supervised learning approach to predict relative depth from monocular images and a transfer learning approach to fine-tune the model on a target dataset with metric depth maps as ground truth. ZoeDepth outperforms state-of-the-art methods on both relative and metric depth estimation tasks on several benchmark datasets and is suitable for real-time applications that require accurate depth estimation at close ranges. Additionally, the proposed method can be trained on multiple datasets, making it suitable for zero-shot transfer scenarios. 

<p id="data"> </p>




## Dataset

### MidAir

MidAir is a large-scale synthetic dataset consisting of over 50,000 photorealistic images and depth maps of indoor and outdoor scenes. The dataset is designed for training and evaluating algorithms for depth estimation and other related tasks in computer vision. The scenes in MidAir are generated using the Unity game engine, which allows for the creation of highly detailed and realistic environments.

Note: For our experiments, we used a subset of the MidAir dataset consisting of 1000 images for training and testing. The images in the dataset are captured from various viewpoints and contain a range of indoor and outdoor scenes, including offices, living rooms, and outdoor environments.

Our MidAir dataset consists of over 1,000 high-resolution monocular images captured from a diverse range of scenes and viewpoints. Here are two sample images of directory of MidAir Dataset that provide a glimpse into the variety of scenes captured:

<img src = "https://user-images.githubusercontent.com/81452190/236362715-4320262d-fbd8-4e6c-8375-9e5b78482208.png" width = "400px">

<img src = "https://user-images.githubusercontent.com/81452190/236362747-0e178560-7d14-4328-8351-e83e5c924a32.png" width = "400px">

### Here are two sample images from the B1 and B2 datasets, which were used for fine-tuning our depth estimation algorithm:

#### B1 Dataset
This is a sample image of a directory of the B1 dataset, which was collected using a drone to fly around Boston University's campus:

<img src = "https://user-images.githubusercontent.com/81452190/236362801-916762d0-cc3d-46c7-849e-33e9e2f4e287.png" width = "400px">

#### B2 Dataset
This is a sample image of a directory of the B2 dataset, which was collected using a drone to fly around Boston University's campus:

<img src = "https://user-images.githubusercontent.com/81452190/236362876-fdfce51a-679f-4441-a60a-b1bedf47819d.png" width = "400px">

These datasets were collected using a drone to fly around Boston University's central campus. BU1 includes footage of Commonwealth Ave. facing west towards the I-90 freeway, while BU2 includes Marsh Chapel and Marsh Plaza footage. We used 1000 images from each of these datasets for fine-tuning our depth estimation algorithm, which resulted in improved performance on the MidAir dataset and other benchmark datasets.

<p id="met"> </p>



## Metrics:

| Models            | RMS      | REL      | ALog     |
|------------------|----------|----------|----------|
| MVS-Net           | 33.179   | 0.327    | 0.124    |
| Monocular Depth   | 22.599   | 0.198    | 0.090    |
| ZoeDepth(KN)      | 108.15   | 21.50    | 0.596    |
| ZoeDepth(NYU)     | 100.78   | 27.00    | 0.624    |
| ZoeDepth(Kitty)   | 111.64   | 22.28    | 0.622    |
| ZoeDepth(SOTA)    | 38.365   | 0.383    | 0.166    |
| Fine-tuned(Ours)  | 21.113   | 0.201    | 0.089    |

Comparing the models' performance in the table, it can be seen that the Fine-tuned model (ours) outperforms all other models in terms of RMS and ALog, achieving the lowest values of 21.113 and 0.089, respectively. In terms of REL, Monocular Depth performs slightly better than our model, but our model is still competitive with a REL value of 0.201. MVS-Net and ZoeDepth models have significantly higher RMS and ALog values, indicating a poorer performance compared to the other models. Overall, our Fine-tuned model performs the best among the models in the comparison.

<p id="res"> </p>



## Result

### MVS-Net 2D Depth Estimation + 3D reconstruction
### Mono-Depth 2D Depth Estimation + 3D reconstruction
### Zoe-Depth 2D Depth Estimation + 3d reconstruction

### NeRF + Depth Estimation:
<img src = "https://user-images.githubusercontent.com/81452190/236475251-7b6f2495-64f8-440c-b509-86481d32de3e.png" width = "500px">

This is the 3D point clouds output + volumn rendered 3D reconstruction for the champel in BU central. 

<p id="ref"> </p>



## References

1. H. C. Longuet-Higgins. A computer algorithm for reconstructing a scene from two projections. Nature, 293:133–135, 1981.

2. J.Sturm, N. Engelhard, F. Endres, W. Burgard, and D. Cremers. RGB-D Mapping: Using Depth Cameras for Dense 3D Modeling of Indoor Environments. In Proceedings of the IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2011.

3. Choi, S. Robust Reconstruction of Indoor Scenes. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2015.

4. Godard, C., Mac Aodha, O., Firman, M., & Brostow, G. (2018). Digging Into Self-Supervised Monocular Depth Estimation. ArXiv. /abs/1806.01260

5. A. Geiger, P. Lenz and R. Urtasun, "Are we ready for autonomous driving? The KITTI vision benchmark suite," 2012 IEEE Conference on Computer Vision and Pattern Recognition, Providence, RI, USA, 2012, pp. 3354-3361, doi: 10.1109/CVPR.2012.6248074.

6. Zhang, J., Yao, Y., Li, S., Luo, Z., & Fang, T. (2020). Visibility-aware Multi-view Stereo Network. ArXiv. /abs/2008.07928

7. Bhat, S. F., Birkl, R., Wofk, D., Wonka, P., & Müller, M. (2023). ZoeDepth: Zero-shot Transfer by Combining Relative and Metric Depth. ArXiv. /abs/2302.12288

8. Mildenhall, B., Srinivasan, P. P., Tancik, M., Barron, J. T., Ramamoorthi, R., & Ng, R. (2020). NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis. ArXiv. /abs/2003.08934

9. Shen, T., Hao, Z., Zhang, R., & Zhang, Y. (2021). Depth-supervised NeRF: Fewer Views and Faster Training for Free.

10. Poornima, S., Lee, K.-H., & Xu, Y. (2019). Mid-Air: A multi-modal dataset for extremely low altitude drone flights.

11. Schönberger, J. L., & Frahm, J.-M. (2016). Structure-from-Motion Revisited. 

12. Zhou, Q. Y., Park, J., & Koltun, V. (2018). Open3D: A Modern Library for 3D Data Processing. arXiv preprint arXiv:1801.09847.

13. SceneNN: A Scene Meshes Dataset with aNNotations.
