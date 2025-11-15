"""
Definición de una red convolucional ligera para clasificación
del dataset ASL Alphabet.

Este módulo incluye:

· La clase SimpleCNN, compuesta por:
   - Cuatro bloques convolucionales con BatchNorm y Mish.
   - Reducción progresiva mediante MaxPooling.
   - Dropout2D tras cada bloque para disminuir el sobreajuste
   - Una capa de Pooling Adaptativo que resume los mapas de activación.
   - Un clasificador final formado por una única capa lineal.

· Instanciación del modelo, junto con:
   - La función de pérdida CrossEntropyLoss.
   - El optimizador AdamW con regularización L2.
"""

from dataset import *
import torch
import torch.nn as nn
import torch.optim as optim

num_classes = n_classes()

# --- Definición de una CNN (Ultra-Ligera) ---
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=num_classes):
        super(SimpleCNN, self).__init__()
        self.relu = nn.Mish()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bloques convolucionales
        self.conv1 = nn.LazyConv2d(out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        
        self.conv2 = nn.LazyConv2d(out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        
        self.conv3 = nn.LazyConv2d(out_channels=32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        
        self.conv4 = nn.LazyConv2d(out_channels=64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        
        self.drop_conv = nn.Dropout2d(0.1) 
        
        # Clasificador
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.3) 
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        # ---- Capa 1 ----
        x = self.conv1(x); x = self.bn1(x); x = self.relu(x); x = self.pool(x)
        x = self.drop_conv(x)

        # ---- Capa 2 ----
        x = self.conv2(x); x = self.bn2(x); x = self.relu(x); x = self.pool(x)
        x = self.drop_conv(x)

        # ---- Capa 3 ----
        x = self.conv3(x); x = self.bn3(x); x = self.relu(x); x = self.pool(x)
        x = self.drop_conv(x)

        # ---- Capa 4 ----
        x = self.conv4(x); x = self.bn4(x); x = self.relu(x); x = self.pool(x)
        x = self.drop_conv(x)
        
        # --- Clasificador ---
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)      
        
        return x

# --- Instanciación ---
model = SimpleCNN(num_classes=num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

if __name__ == '__main__':
    print(f"\n Modelo Ensayo 2 – Modelo CNN ligero (4 bloques: 16-32-32-64)")
    print(f" Número de clases: {num_classes}")
    print(f" Optimizador: AdamW | LR: 0.001 | Dropout: 0.3")
