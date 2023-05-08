import scipy.io
import csv
import cv2

csv_file_name = "../../dataset/labels.csv"
image_path = "../../dataset/raw_images/"

mat =  scipy.io.loadmat('../../raw_data/train_32x32.mat')
images = mat['X']
labels = mat['y']

with open(csv_file_name, 'w', newline='') as csvfile:
    fieldnames = ['Name', 'Label']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    
    # Iterate over all images
    for idx in range(labels.shape[0]):
        image = images[:,:,:,idx]
        label = labels[idx].item()
        image_name = str(idx) + '.jpg'
        
        # Label 0 is used for 10
        if(label == 10):
            label = 0
        
        # Save the image
        print("Extracting " + image_name)
        cv2.imwrite(image_path + image_name, image)

        # Save the label in the CSV file
        row = {'Name'   : image_name,
               'Label'  : label
              }
        writer.writerow(row)
