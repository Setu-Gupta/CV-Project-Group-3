import cv2
import csv

CSV_PATH = "../../dataset/labels.csv"
SRC_DIR = "../../dataset/raw_images/"
DST_DIR = "../../dataset/greyscale_images/"

with open(CSV_PATH, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        image_name = row['Name']

        src_image_path = SRC_DIR + image_name
        print("Processing " + src_image_path)
        
        src_img = cv2.imread(src_image_path)
        dst_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
        
        dst_image_path = DST_DIR + image_name
        cv2.imwrite(dst_image_path, dst_img)
