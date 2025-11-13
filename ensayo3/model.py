# --- ENSAYO 3: CNN Minimalista ---
from dataset import *
import torch
import torch.nn as nn
import torch.optim as optim

num_classes = n_classes()

class MiniCNN(nn.Module):
    def __init__(self, num_classes):
        super(MiniCNN, self).__init__()

        self.act = nn.SiLU()
        self.pool = nn.MaxPool2d(2, 2)

        # --- Bloques convolucionales ---
        self.conv1 = nn.LazyConv2d(32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.LazyConv2d(64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # --- Capa de reducción adaptativa ---
        self.gap = nn.AdaptiveAvgPool2d((2, 2))  # ajusta automáticamente el tamaño de salida

        # --- Clasificador ---
        self.fc1 = nn.Linear(64 * 2 * 2, 128)  # coincide con la salida de GAP (3x3)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.pool(self.act(self.bn1(self.conv1(x))))
        x = self.pool(self.act(self.bn2(self.conv2(x))))

        x = self.gap(x)
        x = torch.flatten(x, 1)

        x = self.dropout(self.act(self.fc1(x)))
        x = self.fc2(x)
        return x


# --- Entrenamiento ---
model = MiniCNN(num_classes=num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.RMSprop(model.parameters(), lr=5e-4, alpha=0.99)

if __name__ == "__main__":
    print(f"\n🧪 Modelo Ensayo 3 – CNN Minimalista (2 bloques: 32-64)")
    print(f"📊 Número de clases: {num_classes}")
    print(f"🔧 Optimizador: RMSprop | LR: 0.001 | Dropout: 0.3 | Label smoothing: 0.1")
