import cv2
import csv
import numpy as np

CSV_PATH = "../../dataset/labels.csv"
SRC_DIR = "../../dataset/greyscale_images/"
DST_DIR = "../../dataset/gabor_horizontal_images/"

# Parameters (handtuned)
ksize = 10
sigma = 1
theta = np.pi/2
lamda = 1.1*np.pi 
gamma = 0.5
phi = 0.5

gabor_kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, phi, ktype=cv2.CV_32F) 

with open(CSV_PATH, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        image_name = row['Name']

        src_image_path = SRC_DIR + image_name
        print("Processing " + src_image_path)
        
        src_img = cv2.imread(src_image_path)
        dst_img = cv2.filter2D(src_img, ddepth=-1, kernel=gabor_kernel)
        
        dst_image_path = DST_DIR + image_name
        cv2.imwrite(dst_image_path, dst_img)
