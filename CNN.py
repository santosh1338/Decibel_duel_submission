import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

class AudioFeatureDataset(Dataset):
    def __init__(self, root_dir, classes=None, is_test=False):
        self.root_dir = root_dir
        self.is_test = is_test
        if not is_test:
            self.classes = classes
            self.filepaths = []
            self.labels = []
            for idx, cls in enumerate(classes):
                cls_folder = os.path.join(root_dir, cls)
                for f in os.listdir(cls_folder):
                    if f.endswith(".wav"):
                        self.filepaths.append(os.path.join(cls_folder, f))
                        self.labels.append(idx)
        else:
            self.filepaths = [os.path.join(root_dir, f) for f in os.listdir(root_dir)]
            self.filepaths.sort()

        self.sample_rate = 16000
        self.mel = torchaudio.transforms.MelSpectrogram(sample_rate=self.sample_rate, n_mels=64)
        self.mfcc = torchaudio.transforms.MFCC(sample_rate=self.sample_rate, n_mfcc=40)

    def __len__(self):
        return len(self.filepaths)

    def load_audio(self, path):
        wav, sr = torchaudio.load(path)
        wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
        return wav

    def __getitem__(self, idx):
        fpath = self.filepaths[idx]
        wav = self.load_audio(fpath)
        mel = self.mel(wav).squeeze(0)
        mfcc = self.mfcc(wav).squeeze(0)
        min_len = min(mel.shape[1], mfcc.shape[1])
        mel = mel[:, :min_len]
        mfcc = mfcc[:, :min_len]
        features = torch.stack([mel, mfcc], dim=0)
        if self.is_test:
            return features, os.path.basename(fpath)
        else:
            return features, self.labels[idx]

class AudioCNN(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return self.fc(x)

classes = ["dog_bark", "drilling", "engine_idling", "siren", "street_music"]

full_dataset = AudioFeatureDataset("Train", classes=classes)
indices = list(range(len(full_dataset)))
train_idx, val_idx = train_test_split(indices, test_size=0.2, shuffle=True, random_state=42)

train_subset = torch.utils.data.Subset(full_dataset, train_idx)
val_subset = torch.utils.data.Subset(full_dataset, val_idx)

train_loader = DataLoader(train_subset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=16, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AudioCNN(num_classes=5).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

epochs = 20

for epoch in range(epochs):
    model.train()
    train_loss = 0
    for X, y in tqdm(train_loader):
        X, y = X.to(device), torch.tensor(y).to(device)
        optimizer.zero_grad()
        preds = model(X)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(device), torch.tensor(y).to(device)
            preds = model(X)
            loss = criterion(preds, y)
            val_loss += loss.item()
            _, predicted = torch.max(preds, 1)
            correct += (predicted == y).sum().item()
            total += y.size(0)

    print(epoch+1, train_loss/len(train_loader), val_loss/len(val_loader), correct/total)

test_dataset = AudioFeatureDataset("Test", is_test=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

model.eval()
results = []

with torch.no_grad():
    for X, fname in test_loader:
        X = X.to(device)
        preds = model(X)
        pred = torch.argmax(preds, dim=1).item()
        label = classes[pred]
        results.append([fname[0], label])

df = pd.DataFrame(results, columns=["ID", "Class"])
df.to_csv("submission.csv", index=False)
