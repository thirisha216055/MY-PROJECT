import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights
from PIL import Image
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from tqdm import tqdm  # NEW: tqdm for progress bar

# ========== CONFIG ==========
IMAGE_DIR = r"C:\Users\HP\Desktop\project2\excel_ds"
BATCH_SIZE = 16  # Reduced for safer training
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
SKIPPED_IMAGE_LOG = "skipped_images.log"

# ========== DATASET ==========
class InteriorDesignDataset(Dataset):
    def __init__(self, csv_path, image_dir, transform=None):
        self.df = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.transform = transform

        self.style_encoder = LabelEncoder()
        self.room_encoder = LabelEncoder()

        self.df['style_encoded'] = self.style_encoder.fit_transform(self.df['style'])
        self.df['room_encoded'] = self.room_encoder.fit_transform(self.df['room_type'])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = os.path.join(self.image_dir, row['image_path'])

        if not os.path.exists(image_path):
            print(f"Missing image: {image_path}")
            with open(SKIPPED_IMAGE_LOG, "a") as f:
                f.write(f"Missing: {image_path}\n")
            return self.__getitem__((idx + 1) % len(self.df))

        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            with open(SKIPPED_IMAGE_LOG, "a") as f:
                f.write(f"Error: {image_path}, {e}\n")
            return self.__getitem__((idx + 1) % len(self.df))

        if self.transform:
            image = self.transform(image)

        style_label = torch.tensor(row['style_encoded'], dtype=torch.long)
        room_label = torch.tensor(row['room_encoded'], dtype=torch.long)

        return image, style_label, room_label

# ========== TRANSFORMS ==========
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ========== LOADERS ==========
train_dataset = InteriorDesignDataset(r"C:\Users\HP\Desktop\project2\excel_ds\train_data.csv", IMAGE_DIR, transform)
val_dataset = InteriorDesignDataset(r"C:\Users\HP\Desktop\project2\excel_ds\val_data.csv", IMAGE_DIR, transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=2)

# ========== MODEL ==========
class RecommenderNet(nn.Module):
    def __init__(self, num_styles, num_rooms):
        super(RecommenderNet, self).__init__()
        self.encoder = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.encoder.fc = nn.Identity()

        self.style_head = nn.Linear(512, num_styles)
        self.room_head = nn.Linear(512, num_rooms)

    def forward(self, x):
        embedding = self.encoder(x)
        style_logits = self.style_head(embedding)
        room_logits = self.room_head(embedding)
        return embedding, style_logits, room_logits

# ========== INIT ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_styles = len(train_dataset.style_encoder.classes_)
num_rooms = len(train_dataset.room_encoder.classes_)

model = RecommenderNet(num_styles, num_rooms).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ========== TRAINING LOOP ==========
def train():
    model.train()
    train_losses = []
    for epoch in range(NUM_EPOCHS):
        total_loss = 0
        for images, style_labels, room_labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            images = images.to(device)
            style_labels = style_labels.to(device)
            room_labels = room_labels.to(device)

            optimizer.zero_grad()
            embeddings, style_logits, room_logits = model(images)

            loss1 = criterion(style_logits, style_labels)
            loss2 = criterion(room_logits, room_labels)
            loss = loss1 + loss2

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {avg_loss:.4f}")

    # Plot training loss
    plt.plot(range(1, NUM_EPOCHS+1), train_losses, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.grid(True)
    plt.savefig("training_loss.png")
    # plt.show()  # Disabled to avoid GUI issues

# ========== PREDICTION EXAMPLE ==========
def predict_single_image(image_path):
    model.eval()
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        _, style_logits, room_logits = model(image)
        style_pred = torch.argmax(style_logits, dim=1).item()
        room_pred = torch.argmax(room_logits, dim=1).item()

    style = train_dataset.style_encoder.inverse_transform([style_pred])[0]
    room = train_dataset.room_encoder.inverse_transform([room_pred])[0]

    print(f"Predicted Style: {style}, Predicted Room Type: {room}")

# ========== RUN ==========
if __name__ == "__main__":
    train()
    torch.save(model.state_dict(), "recommender_model.pth")
    print("Model trained and saved.")

    # Example usage:
    # predict_single_image(r"C:\Users\HP\Desktop\project2\excel_ds\some_image.jpg")
