from torch.utils.data import Dataset
import csv
import cv2

CSV_PATH = "../../../dataset/labels.csv"
SRC_DIR = "../../../dataset/gabor_horizontal_images/"

class GaborHorizontalDataset(Dataset):
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
                
                image_path = SRC_DIR + image_name
                self.X.append(image_path)
                self.Y.append(label)
        
        self.transform = transform

    # Returns the number of samples in the dataset
    def __len__(self):
        return len(self.Y)
    
    # Returns a datapoint and label pair at a given index
    def __getitem__(self, idx):
        image_path = self.X[idx]
        
        # Read the image and the label
        image = cv2.imread(image_path)[:,:,0]
        label = self.Y[idx]
        
        # Apply transform
        if self.transform:
            image = self.transform(image)
        
        return image, label
