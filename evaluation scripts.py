import argparse
import sys
import numpy as np
from PIL import Image
import cv2
import os


def metric(depth_folder, gt_folder):
    # load the path
    depth_files = os.listdir(depth_folder)
    gt_files = os.listdir(gt_folder)

    # Initialize a variable to store the total MSE
    total_rms = 0
    total_are = 0
    total_loge = 0

    for i in range(len(depth_files)):

        # Read the images
        depth_img = cv2.imread(os.path.join(depth_folder, depth_files[i]), cv2.IMREAD_GRAYSCALE)
        # depth_img = cv2.imread(os.path.join(depth_folder, depth_files[i]), cv2.IMREAD_UNCHANGED)
        gt_img = cv2.imread(os.path.join(gt_folder, gt_files[i]), cv2.IMREAD_GRAYSCALE)

        # preprocess
        # scale down the ground truth
        if np.max(depth_img) == np.min(depth_img):
            depth_img = np.max(gt_img)
        else:
            depth_img = (depth_img - np.min(depth_img)) / (np.max(depth_img) - np.min(depth_img))
            depth_img = depth_img * (np.max(gt_img) - np.min(gt_img)) + np.min(gt_img)

        # Calculate the RMS
        mse = np.sqrt(np.mean((depth_img - gt_img) ** 2))

        # Calculate the ARE:
        are = np.mean(np.abs((depth_img - gt_img) / depth_img))

        # Calculate the log:
        log = np.mean(np.abs(np.log10(depth_img) - np.log10(gt_img)))

        total_rms += mse
        total_are += are
        total_loge += log

    avg_rms = total_rms / 1000
    avg_are = total_are / 1000
    avg_log = total_loge / 1000

    print("ROOT MEAN SQUARED ERROR (RMS):", avg_rms)
    print("AVERAGE RELATIVE ERROR (REL):", avg_are)
    print("AVERAGE (LOG10) ERROR:", avg_log)


# Load data path
depth_folder = 'Luis_midair'
gt_folder = 'midair_depth_gt'

metric(depth_folder, gt_folder)