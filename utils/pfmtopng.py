import cv2
import imageio
import glob
import argparse
import shutil
import os
import numpy as np

def main(pfm_path, png_path):
	print("[INFO] PFM_PATH {}    PNG_PATH {}   ".format(pfm_path, png_path))
	pfm_files = glob.glob(pfm_path + '/*.pfm')
	if os.path.exists(png_path):
		shutil.rmtree(png_path)
	os.makedirs(png_path)
	total = len(pfm_files)
	for idx, fpath in enumerate(pfm_files):
		fname = os.path.basename(fpath)
		new_fname = os.path.splitext(fname)[0]
		new_fname += '.png'
		print("[INFO] DEPTH {} to {} ---- {} / {}".format(fname, new_fname, idx + 1, total))
		pfm = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)
		#pfm = np.array(pfm).astype(np.uint16)
		#pfm = np.interp(pfm, (pfm.min(), pfm.max()), (0, 65535))
		print(pfm.dtype)
		#pfm = imageio.v2.imread(fpath)
                #c = 0
		#d = 255
		pfm *= 1000
		print("[INFO] DEPTH MAX {}".format(pfm.max()))
		#depth = (depth - a) * (d - c) / (b - a) + c
		#depth = np.clip(depth, c, d).astype(np.uint8)
		#pfm = pfm / 100
		#pfm = (pfm).astype(np.uint16)
		print("[INFO] DEPTH MAX AFTER DIV 10 {} and dtype {}".format(pfm.max(), pfm.dtype))
		#print("[INFO] depth datatype {}".format(depth.dtype))
		imageio.imwrite(png_path + "/" + new_fname, pfm.astype(np.uint16))
		#cv2.imwrite(png_path + "/" + new_fname, pfm, )
		if idx == 499:
                        print("[info] pfm {}".format(pfm))
		

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='SIMPLE SCRIPT TO CONVERT .PFM TO .PNG FILES')
	parser.add_argument('--pfm', type=str, help='SOURCE PATH TO PFM FILES', required=True)
	parser.add_argument('--png', type=str, help='DESIRED PATH TO PNG FILES', required=True)
	args = parser.parse_args()

	main(args.pfm, args.png)
		
