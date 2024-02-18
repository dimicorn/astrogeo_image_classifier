import torch.nn as nn


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(64*64*64, 128)
        self.fc2 = nn.Linear(128, 7)

        self.relu = nn.ReLU()

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(p=0.2)
    
    def forward(self, image):
        image = self.pool(self.relu(self.conv1(image)))
        image = self.pool(self.relu(self.conv2(image)))
        image = self.pool(self.relu(self.conv3(image)))
        image = image.view(-1, 64*64*64)
        image = self.dropout(self.relu(self.fc1(image)))
        image = self.fc2(image)
        return image