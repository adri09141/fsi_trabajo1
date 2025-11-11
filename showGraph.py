import matplotlib.pyplot as plt

def plot_training_history(history):
    """
    Genera gráficos que muestran la evolución del entrenamiento de un modelo
    a lo largo de las épocas, tanto para la pérdida (loss) como para la 
    precisión (accuracy) en los conjuntos de entrenamiento y validación.

    Args:
        history (dict): Contiene los valores históricos del entrenamiento. 
            Debe tener las siguientes claves:
                'train_loss': lista con los valores de pérdida en entrenamiento.
                'dev_loss': lista con los valores de pérdida en validación.
                'train_acc': lista con los valores de precisión en entrenamiento.
                'dev_acc': lista con los valores de precisión en validación.
    
    
    
    Returns:
        None:  Muestra en pantalla dos gráficos: uno de pérdidas y otro de precisión.            
    """

    # Lista con los números de época
    epochs = range(1, len(history['train_loss']) + 1)

    # Definir el tamaño de la figura(10x4")    plt.figure(figsize=(10, 4))
    plt.figure(figsize=(10, 4))

    # ===========================
    # Gráfico 1: Pérdida (Loss)
    # ===========================
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['dev_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)

    # ================================
    # Gráfico 2: Precisión (Accuracy)
    # ================================
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], label='Train Accuracy')
    plt.plot(epochs, history['dev_acc'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)

    # Ajustar espacios para evitar solapamieno
    plt.tight_layout()

    # Mostrar los gráficos en pantalla
    plt.show()
