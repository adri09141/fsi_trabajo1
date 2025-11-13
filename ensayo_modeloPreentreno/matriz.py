from dataset import val_loader, n_classes
import torch
from torch import nn
from torchvision.models import efficientnet_b0
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # --- Configuración del dispositivo ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    # --- Cargar modelo preentrenado ---
    num_classes = n_classes()
    model = efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    # --- Cargar pesos entrenados ---
    state_dict = torch.load("asl_cnn_final.pth", map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # --- Obtener nombres de clases ---
    try:
        classes = val_loader.dataset.dataset.classes  # si val_loader usa Subset
    except AttributeError:
        classes = val_loader.dataset.classes

    # --- Evaluar en validación ---
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # --- Calcular matriz de confusión ---
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
