import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def init_weights(self):
        torch.nn.init.xavier_uniform_(self.choice.weight.data)
        torch.nn.init.xavier_uniform_(self.conv.weight.data)
        torch.nn.init.xavier_uniform_(self.fc1.weight.data)
        torch.nn.init.xavier_uniform_(self.fc2.weight.data)
        torch.nn.init.xavier_uniform_(self.fc3.weight.data)

    def __init__(self):
        super().__init__()
        self.choice = nn.Conv2d(in_channels=8, out_channels=1, kernel_size=1, padding=0)    # The wighted sum choice layer
        self.conv = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=3, padding=1)     # The convolution layer
        self.fc1 = nn.Linear(10 * 32 * 32, 1000)                                            # First fully connected layer
        self.fc2 = nn.Linear(1000, 100)                                                     # Second fully connected layer
        self.fc3 = nn.Linear(100, 10)                                                       # Last fully connected layer
        self.init_weights()                                                                 # Initialize the weights
        self.softmax = nn.Softmax(dim=1)                                                    # A softmax layer to get predictions

    def forward(self, x):
        x = self.choice(x)                  # Apply the choice layer
        x = self.conv(x)                    # Apply the convolutional layer
        x = F.relu(x)                       # Apply the activation function
        x = torch.flatten(x, start_dim=1)   # Flatten the values to feed the fully connected layer
        x = self.fc1(x)                     # Apply the first fully connected layer
        x = F.relu(x)                       # Apply the activation function
        x = self.fc2(x)                     # Apply the second fully connected layer
        x = F.relu(x)                       # Apply the activation function
        x = self.fc3(x)                     # Apply the final fully connected layer

        return x

    # Predicts probabilities of various classes
    def predict(self, x):
        x = self.forward(x)
        return self.softmax(x)
    
    # Predict the most probable label
    def predict_label(self, x):
        probs = self.predict(x)
        return torch.argmax(probs, dim=1)
    
    # Get the weighted image
    def get_weighted_image(self, x):
        return self.choice(x).reshape(x.shape[1:])
    
    # Get the weights of the choice layer
    def get_weights(self):
        return self.choice.weight
    
    # Get the learnt kernel
    def get_kernel(self):
        return self.conv.weight
