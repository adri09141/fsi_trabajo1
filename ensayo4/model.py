"""
Definición de una red convolucional versión mínima de 5 capas
para clasificación del dataset ASL Alphabet.

Este módulo incluye:

· La clase SimpleCNN_5Layer, compuesta por:
   - Cinco bloques convolucionales con BatchNorm y GELU.
   - Reducción progresiva mediante MaxPooling tras cada bloque.
   - Dropout para reducir el sobreajuste.
   - Una capa de Pooling Adaptativo que resume los mapas de activación.
   - Un clasificador final formado por dos capas lineales.
   
· Instanciación del modelo, junto con:
   - La función de pérdida CrossEntropyLoss.
   - El optimizador Adamax con tasa de aprendizaje 1e-3.
"""

from dataset import *
import torch
import torch.nn as nn
import torch.optim as optim

num_classes = n_classes()

# --- Definición de una CNN de 5 Capas minimalista ---
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN_5Layer, self).__init__()

        self.act = nn.GELU() 
        self.pool = nn.MaxPool2d(2, 2)

        # Bloques convolucionales
        self.conv1 = nn.LazyConv2d(1, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(1)
        
        self.conv2 = nn.LazyConv2d(2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(2)
        
        self.conv3 = nn.LazyConv2d(4, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(4)
        
        self.conv4 = nn.LazyConv2d(8, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(8)
        
        self.conv5 = nn.LazyConv2d(16, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(16)

        # Clasificador
        self.gap = nn.AdaptiveAvgPool2d((1, 1)) 
        self.fc1 = nn.LazyLinear(128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.2) 

    def forward(self, x):
        x = self.pool(self.act(self.bn1(self.conv1(x))))
        x = self.pool(self.act(self.bn2(self.conv2(x))))
        x = self.pool(self.act(self.bn3(self.conv3(x)))) 
        x = self.pool(self.act(self.bn4(self.conv4(x)))) 
        x = self.pool(self.act(self.bn5(self.conv5(x)))) 

        x = self.gap(x)
        x = torch.flatten(x, 1)

        x = self.dropout(self.act(self.fc1(x)))
        x = self.fc2(x)
        
        return x

# --- Entrenamiento ---
model = SimpleCNN_5Layer(num_classes=num_classes)
criterion = nn.CrossEntropyLoss() 
optimizer = optim.Adamax(model.parameters(), lr=1e-3)

if __name__ == "__main__":
    print(f"\nModelo Ensayo 4 – CNN de 5 Capas (¡VERSIÓN MÍNIMA con Adamax!)")
    print(f"Número de clases: {num_classes}")
    print(f"Optimizador: Adamax | LR: 1e-3 | Dropout: 0.2")
