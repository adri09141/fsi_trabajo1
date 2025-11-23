"""
Script para analizar un checkpoint de entrenamiento:
- Carga 'asl_cnn_checkpoint.pth'
- Muestra métricas promedio
- Dibuja gráficas de pérdida y accuracy
- Compara las medias mediante gráficos simples y claros.

Requisitos:
    torch, matplotlib
"""

import torch
import matplotlib.pyplot as plt
import numpy as np

# ====================================================
# Gráficas de loss y accuracy por separado
# ====================================================
def plot_history(history):

    epochs = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(10,5))
    plt.plot(epochs, history['train_loss'], label="Train Loss")
    plt.plot(epochs, history['dev_loss'], label="Validation Loss")
    plt.title("Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10,5))
    plt.plot(epochs, history['train_acc'], label="Train Accuracy")
    plt.plot(epochs, history['dev_acc'], label="Validation Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# ====================================================
# Barras simples para comparar medias
# ====================================================
def plot_bar_means(mean_train_loss, mean_dev_loss, mean_train_acc, mean_dev_acc):

    labels = ["Train Loss", "Dev Loss", "Train Acc", "Dev Acc"]
    values = [mean_train_loss, mean_dev_loss, mean_train_acc, mean_dev_acc]
    colors = ['skyblue', 'dodgerblue', 'lightcoral', 'indianred']

    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, values, color=colors, alpha=0.9)

    # Añadir valores encima
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2,
                 height + 0.01,
                 f'{height:.2f}',
                 ha='center',
                 fontsize=12)

    plt.title("Mean Metrics Comparison", fontsize=14)
    plt.ylabel("Value")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()



# ====================================================
# Línea simple para las medias
# ====================================================
def plot_line_means(means):

    labels = ["Train Loss", "Dev Loss", "Train Acc", "Dev Acc"]

    plt.figure(figsize=(10, 6))
    plt.plot(labels, means, marker='o', linewidth=2)

    # Añadir valores encima
    for i, value in enumerate(means):
        plt.text(i, value + 0.01, f"{value:.2f}", ha='center', fontsize=12)

    plt.title("Mean Metrics Comparison", fontsize=14)
    plt.ylabel("Value")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


# =================
# BLOQUE PRINCIPAL
# =================
if __name__ == '__main__':

    checkpoint_path = 'asl_cnn_checkpoint.pth' # Cambiar nombre para cada ensayo
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    except FileNotFoundError:
        raise SystemExit(f"\nERROR: No se encontró el checkpoint: {checkpoint_path}\n")

    history = checkpoint['history']

    train_loss = history['train_loss']
    dev_loss = history['dev_loss']
    train_acc = history['train_acc']
    dev_acc = history['dev_acc']

    # Medias
    mean_train_loss = sum(train_loss) / len(train_loss)
    mean_dev_loss   = sum(dev_loss) / len(dev_loss)
    mean_train_acc  = sum(train_acc) / len(train_acc)
    mean_dev_acc    = sum(dev_acc) / len(dev_acc)

    print("\n===== MÉTRICAS PROMEDIO =====")
    print(f"Mean Train Loss: {mean_train_loss:.4f}")
    print(f"Mean Dev Loss:   {mean_dev_loss:.4f}")
    print(f"Mean Train Acc:  {mean_train_acc:.2f}%")
    print(f"Mean Dev Acc:    {mean_dev_acc:.2f}%\n")

    means = [
        mean_train_loss,
        mean_dev_loss,
        mean_train_acc,
        mean_dev_acc
    ]

    plot_history(history)
    plot_bar_means(*means)
    plot_line_means(means)   
