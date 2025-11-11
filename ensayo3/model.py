from dataset import *
import torch
import torch.nn as nn
import torch.optim as optim

num_classes = n_classes()

# --- Definición de una CNN (Versión Ultra-Ligera) ---
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=num_classes):
        super(SimpleCNN, self).__init__()
        self.relu = nn.Mish()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 🔹 Cuerpo Convolucional (16 -> 32 -> 32 -> 64)
        self.conv1 = nn.LazyConv2d(out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.LazyConv2d(out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.LazyConv2d(out_channels=32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.LazyConv2d(out_channels=64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        # 🔹 Clasificador
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.3) 
        # Una sola capa lineal de 64 (de conv4) a las clases
        self.fc = nn.Linear(64, num_classes)
        # ----------------------------------------------------

    def forward(self, x):
        # --- Cuerpo Conv ---
        x = self.conv1(x); x = self.bn1(x); x = self.relu(x); x = self.pool(x)
        x = self.conv2(x); x = self.bn2(x); x = self.relu(x); x = self.pool(x)
        x = self.conv3(x); x = self.bn3(x); x = self.relu(x); x = self.pool(x)
        x = self.conv4(x); x = self.bn4(x); x = self.relu(x); x = self.pool(x)
        # --- Clasificador ---
        x = self.gap(x)
        x = torch.flatten(x, 1) # Shape: [B, 64]
        x = self.dropout(x)
        # Directo a la capa final
        x = self.fc(x)          # [B, 64] -> [B, num_classes]
        return x

# --- Instanciación ---
model = SimpleCNN(num_classes=num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-4)

if __name__ == '__main__':
    print(f"\n Modelo CNN (Ultra-Ligero 16-32-32-64) con {num_classes} clases.")
