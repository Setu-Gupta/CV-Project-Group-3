import cv2
import csv
import numpy as np

CSV_PATH = "../../dataset/labels.csv"
SRC_DIR = "../../dataset/greyscale_images/"
DST_DIR = "../../dataset/canny_images/"

# Parameters (handtuned)
thresh1 = 100 
thresh2 = 150

with open(CSV_PATH, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        image_name = row['Name']

        src_image_path = SRC_DIR + image_name
        print("Processing " + src_image_path)
        
        src_img = cv2.imread(src_image_path)
        dst_img = cv2.Canny(src_img, thresh1, thresh2)
        
        dst_image_path = DST_DIR + image_name
        cv2.imwrite(dst_image_path, dst_img)
