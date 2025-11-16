"""
Script para generar y visualizar la matriz de confusión de un modelo entrenado
(EfficientNet-B0) aplicado al conjunto de validación del dataset de ASL.
    
    1. Detecta y selecciona automáticamente el dispositivo de ejecución 
       (GPU si está disponible, de lo contrario CPU).
    2. Carga el modelo EfficientNet-B0 preentrenado y ajusta la última capa 
       para el número de clases del dataset.
    3. Carga los pesos entrenados desde 'asl_cnn_final.pth' y pone el modelo 
       en modo evaluación.
    4. Obtiene los nombres de las clases directamente desde la estructura del 
       dataset de validación, adaptándose si val_loader es un Subset.
    5. Recorre todo el conjunto de validación para obtener:
         - Etiquetas verdaderas
         - Predicciones del modelo
       ·Usando torch.no_grad() para evitar el cálculo de gradientes y acelerar la inferencia.
    6. Calcula la matriz de confusión con sklearn y la muestra mediante un mapa 
       de calor usando seaborn, permitiendo identificar qué clases se confunden.
"""

from dataset import val_loader, n_classes
import torch
from torch import nn
from torchvision.models import efficientnet_b0
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Dispositivo de ejecución (GPU si está disponible)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    # --- Cargar modelo EfficientNet-B0 y ajustar última capa ---
    num_classes = n_classes()
    model = efficientnet_b0(weights=None) # No se cargan pesos preentrenados
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    # --- Cargar pesos entrenados ---
    state_dict = torch.load("asl_cnn_final.pth", map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Obtener los nombres de las clases del dataset
    try:
        classes = val_loader.dataset.dataset.classes  # si val_loader usa Subset
    except AttributeError:
        classes = val_loader.dataset.classes

    all_preds, all_labels = [], []

    # ----- Inferencia en el conjunto de validación -----
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # ----- Matriz de confusión -----
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=classes,
                yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix – EfficientNet-B0")
    plt.tight_layout()
    plt.show()
