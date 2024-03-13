import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import numpy as np
import ast

class CustomDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform
        

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = self.dataframe.iloc[idx, 0]
        coordinates = self.dataframe.iloc[idx, 1:].values.astype(float)
        image = Image.open(img_name)
        
        if self.transform:
            image = self.transform(image)
        
        return image , coordinates.astype(float)

# Assuming your DataFrame has one column for image paths and four columns for coordinates: 'image_path', 'x1', 'y1', 'x2', 'y2'
dfs = pd.read_excel(r'C:\Users\Shri\train0.xlsx') 
def change_type(x):
    return list(ast.literal_eval(x))
dfs['Changed_coordinates'] = dfs['Agent Coordinate'].apply(change_type)
dfs[['x1', 'y1']] = dfs['Changed_coordinates'].apply(lambda x: pd.Series([x[0], x[1]]))
df=pd.DataFrame([dfs.Path, dfs.x1, dfs.y1]).transpose()

#pd.read_csv('path_to_your_dataframe.csv')

# Define transforms for data preprocessing
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create custom dataset
custom_dataset = CustomDataset(dataframe=df, transform=transform)

# Create data loader
train_loader = DataLoader(custom_dataset, batch_size=32, shuffle=True)

# Load pre-trained ResNet model
resnet_model = models.resnet50(pretrained=True)

# Define new output layer for regression (assuming coordinates)
resnet_model.fc = nn.Linear(resnet_model.fc.in_features, 2)  # Assuming 4 coordinates (x1, y1, x2, y2)

# Define loss function for regression
criterion = nn.L1Loss()

# Define optimizer
optimizer = optim.Adam(resnet_model.parameters(), lr=0.001)

# Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet_model.to(device)
num_epochs = 10

for epoch in range(num_epochs):
    resnet_model.train()
    running_loss = 0.0
    for images, coordinates in train_loader:
        images, coordinates = images.to(device), coordinates.to(device)
        
        optimizer.zero_grad()
        
        outputs = resnet_model(images)
        #print(type(outputs))
        #print(type(coordinates))
        loss = criterion(outputs, coordinates)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        
    epoch_loss = running_loss / len(custom_dataset)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

# Save the trained model
torch.save(resnet_model.state_dict(), 'resnet_custom_dataset.pth')
