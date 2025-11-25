"""
Script para generar y visualizar la matriz de confusión de un modelo entrenado
(SimpleCNN) aplicado al conjunto de validación del dataset de ASL.
    
    1. Detecta y selecciona automáticamente el dispositivo de ejecución 
       (GPU si está disponible, de lo contrario CPU).
    2. Carga el modelo previamente entrenado desde 'asl_cnn_final.pth' y lo
       configura en modo evaluación.
    3. Obtiene los nombres de las clases directamente desde la estructura del
       dataset de validación.
    4. Recorre todo el conjunto de validación para obtener:
         - Etiquetas verdaderas
         - Predicciones del modelo
       ·Usando torch.no_grad() para evitar el cálculo de gradientes y acelerar la inferencia.
    5. Construye la matriz de confusión 
    6. Muestra la matriz mediante un mapa de calor usando seaborn, facilitando 
       la visualización de qué clases se confunden más entre sí.
"""

from dataset import val_loader, n_classes
from model import SimpleCNN_5Layer as SimpleCNN
import torch
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # Dispositivo de ejecución (GPU si está disponible)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----- Cargar modelo entrenado -----
    num_classes = n_classes()
    model = SimpleCNN(num_classes=num_classes)
    model.load_state_dict(torch.load("asl_cnn_final.pth", map_location=device))
    model.to(device)
    model.eval()

    # Obtener los nombres de las clases del dataset
    classes = val_loader.dataset.dataset.classes

    all_preds = []
    all_labels = []
    
    # ----- Inferencia en el conjunto de validación -----
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # ----- Matriz de confusión -----
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
