import cv2
import csv
import numpy as np

CSV_PATH = "../../dataset/labels.csv"
SRC_DIR = "../../dataset/greyscale_images/"
DST_DIR = "../../dataset/sobelxy_images/"

# Parameters:
GAUSSIAN_BLURR_SIGMA = 3
KERNEL_SIZE = 3


with open(CSV_PATH, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        image_name = row['Name']

        src_image_path = SRC_DIR + image_name
        print("Processing " + src_image_path)
        
        src_img = cv2.imread(src_image_path)
        img_blur = cv2.GaussianBlur(src_img, (GAUSSIAN_BLURR_SIGMA, GAUSSIAN_BLURR_SIGMA), 0)
        dst_img = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=KERNEL_SIZE)
	
        dst_image_path = DST_DIR + image_name
        cv2.imwrite(dst_image_path, dst_img)
