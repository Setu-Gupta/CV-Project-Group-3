from torch.utils.data import Dataset
import numpy as np
import csv
import cv2

CSV_PATH = "../../dataset/labels.csv"
SRC_DIR = "../../dataset/"
DATASETS = ["canny_images/",
            "gabor_horizontal_images/",
            "gabor_vertical_images/",
            "greyscale_images/",
            "laplacian_images/",
            "sobelx_images/",
            "sobely_images/",
            "sobelxy_images/"
            ]

class CombinedDataset(Dataset):
    # Loads the data and populates the X and Y variables
    def __init__(self, transform=None):
        
        # Save the image path and the labels
        self.X = []
        self.Y = []
        with open(CSV_PATH, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                image_name = row['Name']
                label = int(row['Label'])
                
                self.X.append([])
                for dataset in DATASETS:
                    image_path = SRC_DIR + dataset + image_name
                    self.X[-1].append(image_path)
                self.Y.append(label)
        
        self.transform = transform

    # Returns the number of samples in the dataset
    def __len__(self):
        return len(self.Y)
    
    # Returns a datapoint and label pair at a given index
    def __getitem__(self, idx):
        # Read the images and the label
        label = self.Y[idx]
        path_list = self.X[idx]
        image = cv2.imread(path_list[0])[:,:,0]
        image = np.expand_dims(image, axis=-1)
        for path in path_list[1:]:
            image_new = cv2.imread(path)[:,:,0]
            image_new = np.expand_dims(image_new, axis=-1)
            image = np.append(image, image_new, axis=-1)

        # Apply transform
        if self.transform:
            image = self.transform(image)
        
        return image, label

    def dump_images(self, weights, path):
        for image_paths in self.X:
            image_name = image_paths[0].replace(SRC_DIR, "").replace(DATASETS[0], "")
            image_path = path + image_name
            
            image = cv2.imread(image_paths[0]) * weights[0]
            for img_pth, w in zip(image_paths[1:], weights[1:]):
                image += cv2.imread(img_pth) * w

            cv2.imwrite(image_path, image)
