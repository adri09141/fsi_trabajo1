from dataset import *
import torch
from torch import nn, optim
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

# ------------------------------------------------------------
# Cargar modelo preentrenado (ResNet50 con pesos de ImageNet)
# ------------------------------------------------------------
weights = EfficientNet_B0_Weights.IMAGENET1K_V1  # Usa la versión más nueva
model = efficientnet_b0(weights=weights)

# Congelar todas las capas (para entrenar solo la última)
for param in model.parameters():
    param.requires_grad = False

# Reemplazar la capa final (fc) para adaptarla a tus clases
num_classes = n_classes()  # función que ya tienes definida
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, num_classes)

# Definir criterio y optimizador solo para la capa fc
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.classifier[1].parameters(), lr=1e-3)

if __name__ == '__main__':
    print(f"\nResNet50 preentrenado listo para entrenar con {num_classes} clases.")
