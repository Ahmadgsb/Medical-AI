import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import pickle

# Load unique medicines
with open('unique_meds.pkl', 'rb') as f:
    unique_meds = pickle.load(f)

print(f"Loading {len(unique_meds)} medicines")

# Dataset class
class PrescriptionDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform):
        self.df = pd.read_csv(csv_path, encoding='latin1')
        self.df['medicine_name'] = self.df['medicine_name'].apply(lambda x: [] if x == '__' else [m.strip() for m in x.split(',')])
        self.img_dir = img_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]['image_name']
        img_path = f"{self.img_dir}/{img_name}"
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        
        medicines = self.df.iloc[idx]['medicine_name']
        labels = [1 if m.lower().strip() in [x.lower().strip() for x in medicines] else 0 for m in unique_meds]
        return image, torch.tensor(labels, dtype=torch.float32)

# Simple model
class SimpleModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        from torchvision import models
        self.backbone = models.resnet50(pretrained=True)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return torch.sigmoid(self.backbone(x))

# Training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

dataset = PrescriptionDataset('train_labels.csv', 'Prescriptions Dataset/train', transform)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

model = SimpleModel(len(unique_meds))
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("Training started...")
for epoch in range(5):
    total_loss = 0
    for images, labels in loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/5 | Loss: {total_loss/len(loader):.4f}")

torch.save(model.state_dict(), 'model.pth')
print("Model saved as model.pth")