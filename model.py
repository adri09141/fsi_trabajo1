from dataset import *
import torch
import torch.nn as nn
import torch.optim as optim

num_classes = n_classes()

# --- Definición de una CNN simple y clara ---
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=num_classes):
        super(SimpleCNN, self).__init__()
        self.relu = nn.ReLU()                      # Función de activación no lineal
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Reduce a la mitad alto/ancho
        self.dropout = nn.Dropout(0.4)             # Evita sobreajuste

        # 🔹 Capa 1: Conv -> BatchNorm -> ReLU -> Pool
        # Entrada: [B, 3, 128, 128]  →  Salida: [B, 16, 64, 64]
        self.conv1 = nn.LazyConv2d(out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)

        # 🔹 Capa 2: Conv -> BatchNorm -> ReLU -> Pool
        # Entrada: [B, 16, 64, 64]  →  Salida: [B, 32, 32, 32]
        self.conv2 = nn.LazyConv2d(out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        # 🔹 Capa 3: Conv -> BatchNorm -> ReLU -> Pool
        # Entrada: [B, 32, 32, 32]  →  Salida: [B, 64, 16, 16]
        self.conv3 = nn.LazyConv2d(out_channels=64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        # 🔹 Capa totalmente conectada (clasificación)
        # Aplanamos: [B, 64, 8, 8] → [B, 64*16*16 = 16384]
        self.fc1 = nn.LazyLinear(out_features=1024)
        self.bn_fc1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 256)
        self.bn_fc2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        # ---- Capa 1 ----
        x = self.conv1(x)              # [B, 3, 128, 128] → [B, 16, 128, 128]
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)               # [B, 16, 64, 64]  (↓ tamaño a la mitad)

        # ---- Capa 2 ----
        x = self.conv2(x)              # [B, 16, 64, 64] → [B, 32, 64, 64]
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)               # [B, 32, 32, 32]

        # ---- Capa 3 ----
        x = self.conv3(x)              # [B, 32, 32, 32] → [B, 64, 32, 32]
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool(x)               # [B, 64, 16, 16]

        # ---- Clasificador ----
        x = torch.flatten(x, 1)        # [B, 64, 16, 16] → [B, 4096]
        x = self.dropout(self.relu(self.bn_fc1(self.fc1(x))))   # [B, 4096] → [B, 256]
        x = self.dropout(self.relu(self.bn_fc2(self.fc2(x))))
        x = self.fc3(x) # [B, 256] → [B, num_classes]
        return x

# --- Instanciación ---
model = SimpleCNN(num_classes=num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
if __name__ == '__main__':
    print(f"\n Modelo CNN con {num_classes} clases listo para entrenar.")
