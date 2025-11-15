"""
Definición de una red neuronal convolucional simple para clasificación
del dataset ASL Alphabet.

Este módulo incluye:

· La clase SimpleCNN, una arquitectura CNN compuesta por:
   - Tres bloques de convolución con BatchNorm, ReLU y MaxPooling.
   - Capas fully connected para la clasificación final.
   - Dropout y Dropout2d para reducir sobreajuste.
   - Capas perezosas (LazyConv2d y LazyLinear) que infieren automáticamente
     el tamaño de entrada.

· Instanciación del modelo, junto con:
   - La función de pérdida CrossEntropyLoss.
   - El optimizador Adam con regularización L2.
"""

from dataset import *
import torch
import torch.nn as nn
import torch.optim as optim

num_classes = n_classes()

# --- Definición de una CNN simple y clara ---
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=num_classes):
        super(SimpleCNN, self).__init__()´

        self.relu = nn.ReLU()                    
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  
        self.dropout = nn.Dropout(0.3)            

        # Bloques convolucionales
        self.conv1 = nn.LazyConv2d(out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.LazyConv2d(out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.LazyConv2d(out_channels=64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.drop_conv = nn.Dropout2d(0.1)

        # Capas fully-connected
        self.fc1 = nn.LazyLinear(out_features=1024)
        self.bn_fc1 = nn.BatchNorm1d(1024)

        self.fc2 = nn.Linear(1024, 256)
        self.bn_fc2 = nn.BatchNorm1d(256)

        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        # ---- Capa 1 ----
        x = self.conv1(x)             
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)              

        # ---- Capa 2 ----
        x = self.conv2(x)             
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)              
        x = self.drop_conv(x)

        # ---- Capa 3 ----
        x = self.conv3(x)            
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool(x)              
        x = self.drop_conv(x)

        # ---- Clasificador ----
        x = torch.flatten(x, 1)        
        x = self.dropout(self.relu(self.bn_fc1(self.fc1(x))))   
        x = self.dropout(self.relu(self.bn_fc2(self.fc2(x))))
        x = self.fc3(x) 

        return x

# --- Instanciación ---
model = SimpleCNN(num_classes=num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

if __name__ == '__main__':
    print(f"\n Modelo CNN con {num_classes} clases listo para entrenar.")
