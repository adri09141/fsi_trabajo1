
from dataset import *
import torch
import torch.nn as nn
import torch.optim as optim

num_classes = n_classes()

# --- Definición de una CNN simple y clara ---
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=num_classes):
        super(SimpleCNN, self).__init__()
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.3)

        # 🔹 Capa 1: [B, 3, 96, 96]  →  [B, 8, 48, 48]
        self.conv1 = nn.LazyConv2d(out_channels=8, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)

        # 🔹 Capa 2: [B, 8, 48, 48]  →  [B, 16, 24, 24]
        self.conv2 = nn.LazyConv2d(out_channels=16, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)

        # 🔹 Capa 3: [B, 16, 24, 24]  → [B, 32, 12, 12]
        self.conv3 = nn.LazyConv2d(out_channels=32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        
        # 🔹 Capa 4 (NUEVA): [B, 32, 12, 12] → [B, 64, 6, 6]
        self.conv4 = nn.LazyConv2d(out_channels=64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)

        # 🔹 Clasificador (¡Ahora recibe 64 canales!)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes) # <-- 64 de la última capa


    def forward(self, x):
        x = self.conv1(x); x = self.bn1(x); x = self.relu(x); x = self.pool(x)
        x = self.conv2(x); x = self.bn2(x); x = self.relu(x); x = self.pool(x)
        x = self.conv3(x); x = self.bn3(x); x = self.relu(x); x = self.pool(x)
        x = self.conv4(x); x = self.bn4(x); x = self.relu(x); x = self.pool(x) # <-- NUEVA CAPA
        
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

# --- Instanciación ---
model = SimpleCNN(num_classes=num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-4)
if __name__ == '__main__':
    print(f"\n Modelo CNN con {num_classes} clases listo para entrenar.")
