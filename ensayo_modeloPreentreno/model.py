"""
Configuración de EfficientNet-B0 preentrenado para clasificación
del dataset ASL Alphabet.

Este módulo incluye:

· Carga de EfficientNet-B0 con pesos de ImageNet.
· Congelamiento de todas las capas excepto los últimos bloques y la capa classifier.
· Reemplazo de la capa final para adaptarla al número de clases del dataset.
· Instanciación del modelo, junto con:
    - La función de pérdida CrossEntropyLoss.
    - El optimizador AdamW para la capa final.
"""

from dataset import *
import torch
from torch import nn, optim
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

# ------------------------------------------------------------
# Cargar modelo preentrenado (ResNet50 con pesos de ImageNet)
# ------------------------------------------------------------
weights = EfficientNet_B0_Weights.IMAGENET1K_V1 
model = efficientnet_b0(weights=weights)

# -------------------------------------------------------------------
# Congelar todas las capas excepto los últimos bloques y classifier
# -------------------------------------------------------------------
for name, param in model.named_parameters():
    param.requires_grad = False
    if "features.7" in name or "features.8" in name or "classifier" in name:
        param.requires_grad = True 

# ---------------------------------------------------
# Reemplazar la capa final para el número de clases
# ---------------------------------------------------
num_classes = n_classes()  
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, num_classes)

# ---------------------------------------------------------
# Definir criterio y optimizador (solo para la capa final)
# ---------------------------------------------------------
criterion = nn.CrossEntropyLoss()
params_to_update = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.AdamW(params_to_update, lr=1e-4)

if __name__ == '__main__':
    print(f"\nResNet50 preentrenado listo para entrenar con {num_classes} clases.")
