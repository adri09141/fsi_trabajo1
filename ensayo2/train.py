"""
Script de entrenamiento, validación y evaluación para un modelo CNN.

Este archivo realiza las siguientes tareas:
    - Entrena un modelo definido externamente usando batches del DataLoader.
    - Evalúa el desempeño del modelo en un conjunto de validación en cada época.
    - Ajusta automáticamente el learning rate mediante ReduceLROnPlateau.
    - Registra métricas como pérdida y exactitud durante entrenamiento/validación.
    - Identifica los mejores valores de accuracy para train y validación.
    - Guarda el modelo final una vez terminado el entrenamiento.
    - Evalúa el modelo entrenado en el conjunto de test.

Requiere los módulos:
    - dataset.py: contiene train_loader, val_loader, test_loader y n_classes()
    - model.py: contiene model, criterion y optimizer ya configurados
    - showGraph.py: contiene plot_training_history()
"""

import torch
import torch.nn.functional as F
# --- Importaciones desde tus otros archivos ---
from dataset import train_loader, val_loader, n_classes, test_loader
from model import model, criterion, optimizer  # Traemos el modelo, la función de pérdida y el optimizador ya definidos
from showGraph import plot_training_history
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time

# =====================
# Función: evaluate()
# =====================
def evaluate(model, test_loader):
    """
    Evalúa un modelo en un conjunto de datos (test o validación).

    Recorre todos los batches del DataLoader, realiza inferencia sin 
    cálculo de gradientes y obtiene la cantidad total de predicciones 
    correctas para calcular el porcentaje de exactitud.

    Args:
        model (torch.nn.Module): Modelo a evaluar.
        test_loader (torch.utils.data.DataLoader): DataLoader con el conjunto de datos.

    Returns:
        float: Accuracy expresada como porcentaje.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval() 
    correct = 0
    total = 0
    with torch.no_grad():  
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)  # Toma la clase con mayor probabilidad
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total if total > 0 else 0.0
    print(f'Evaluación - Total: {total}, Correctos: {correct}')  # Info útil para revisar
    return accuracy


# ==================================
# Función: train_with_validation()
# ==================================
def train_with_validation(model, train_loader, dev_loader, criterion, optimizer, epochs=5):
    """
    Entrena el modelo durante 'epochs' épocas usando el conjunto de entrenamiento,
    y evalúa su desempeño en validación al final de cada una.

    Args:
        model (torch.nn.Module): Modelo que se entrenará.
        train_loader (torch.utils.data.DataLoader): Datos de entrenamiento.
        dev_loader (torch.utils.data.DataLoader): Datos de validación.
        criterion (torch.nn.Module): Función de pérdida.
        optimizer (torch.optim.Optimizer): Optimizador del modelo.
        epochs (int): Cantidad total de épocas a entrenar.

    Returns:
        tuple:
            model (torch.nn.Module): Modelo ya entrenado.
            history (dict): Historial de métricas con formato:
                {
                    'train_loss': [...],
                    'train_acc': [...],
                    'dev_loss': [...],
                    'dev_acc': [...]
                }
    """
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',       # Queremos minimizar la pérdida
        factor=0.5,       # Reduce el LR a la mitad
        patience=1       # Espera 2 épocas sin mejora
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Entrenando en dispositivo: {device}")
    model.to(device)

    # Diccionario para ir guardando la evolución del entrenamiento
    history = {
        'train_loss': [],
        'train_acc': [],
        'dev_loss': [],
        'dev_acc': [],
    }

    count_bad_epochs = 0
    best_acc_train, epoch_best_acc_train = 0.0, 0
    best_acc_val, epoch_best_acc_val = 0.0, 0
    
    num_classes = n_classes()  # Obtenemos la cantidad de clases
    epoch_times = []
    total_start_time = time.time()

    # ------------------ Bucle de entrenamiento ------------------
    for epoch in range(epochs):
        start_time = time.time()
        model.train()  # Modo entrenamiento
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        # Entrenamiento por batches
        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()  # Limpiamos gradientes de la iteración anterior
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)  # CrossEntropyLoss espera índices de clase

            loss.backward()  # Backpropagation
            optimizer.step()  # Actualizamos los pesos

            running_loss += loss.item()

            # Calculamos la precisión del batch
            _, predicted_train = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted_train == labels).sum().item()

            # Mostrar progreso cada cierto número de batches
            if (i + 1) % 50 == 0:
                print(f'  [Época {epoch + 1}, Batch {i + 1:4d}] Pérdida entrenamiento: {loss.item():.4f}')

        # Promedio de pérdida y accuracy por época
        avg_train_loss = running_loss / len(train_loader)
        train_acc = 100.0 * correct_train / total_train if total_train > 0 else 0.0

        # ------------------- Validación -------------------
        model.eval()
        dev_running_loss = 0.0
        correct_dev = 0
        total_dev = 0

        with torch.no_grad():
            for inputs, labels in dev_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                dev_loss = criterion(outputs, labels)
                dev_running_loss += dev_loss.item()

                _, predicted_dev = torch.max(outputs.data, 1)
                total_dev += labels.size(0)
                correct_dev += (predicted_dev == labels).sum().item()

        avg_dev_loss = dev_running_loss / len(dev_loader)
        dev_acc = 100.0 * correct_dev / total_dev if total_dev > 0 else 0.0

        # Guardamos los resultados de la época
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        history['dev_loss'].append(avg_dev_loss)
        history['dev_acc'].append(dev_acc)

        # Tiempo por época
        epoch_time = time.time() - start_time
        epoch_times.append(epoch_time)
        mins, secs = divmod(epoch_time, 60)

        # Mostramos resumen de la época
        print(f'\n--- Época {epoch + 1}/{epochs} completada ---')
        print(f'Pérdida Entrenamiento : {avg_train_loss:.4f} | Exactitud Entrenamiento : {train_acc:.2f}%')
        print(f'Pérdida Validación    : {avg_dev_loss:.4f} | Exactitud Validación    : {dev_acc:.2f}%\n')
        print(f'Tiempo por época      : {int(mins)}m {int(secs)}s%\n')

        scheduler.step(avg_dev_loss)
        torch.cuda.empty_cache() 

        # (Opcional) Mostrar el LR actual
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Learning rate actual: {current_lr:.6f}")
        print(f"Épocas sin mejora (num_bad_epochs): {scheduler.num_bad_epochs}\n")

        # Actualizamos los mejores accuracies si es necesario
        if train_acc > best_acc_train:
            best_acc_train = train_acc
            epoch_best_acc_train = epoch + 1
        if dev_acc > best_acc_val:
            best_acc_val = dev_acc
            epoch_best_acc_val = epoch + 1
        if scheduler.num_bad_epochs > 0:
            count_bad_epochs += 1

    total_time = time.time() - total_start_time
    avg_epoch_time = sum(epoch_times) / len(epoch_times)
    total_mins, total_secs = divmod(total_time, 60)
    avg_mins, avg_secs = divmod(avg_epoch_time, 60)

    print("\n--- Entrenamiento finalizado ---")
    print(f"Tiempo total de entrenamiento : {int(total_mins)}m {int(total_secs)}s")
    print(f"Tiempo promedio por época     : {int(avg_mins)}m {int(avg_secs)}s")
    print(f"Mejor Exactitud en Entrenamiento : {best_acc_train:.2f} en la epoca {epoch_best_acc_train}")
    print(f"Mejor Exactitud en Validación     : {best_acc_val:.2f}% en la epoca {epoch_best_acc_val}")
    print(f"Número de épocas sin mejora     : {count_bad_epochs}")
    print("-----------------------------------\n")

    return model, history


# ======================================
# Bloque principal — aquí arranca todo
# ======================================
if __name__ == '__main__':
    num_epochs_to_train = 20  # Definir cuántas épocas quieres entrenar
    print(f'\n--- Iniciando entrenamiento por {num_epochs_to_train} épocas ---')

    trained_model, training_history = train_with_validation(
        model=model,
        train_loader=train_loader,
        dev_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        epochs=num_epochs_to_train
    )

    # Graficamos la evolución de pérdida y accuracy
    plot_training_history(training_history)
    
    print("Guardando el modelo entrenado...")
    torch.save(trained_model.state_dict(), 'asl_cnn_final.pth')

    # Esto es para guardar el modelo, para futuras métricas
    
    torch.save({
        'epoch': num_epochs_to_train,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': training_history
    }, 'asl_cnn_checkpointEnsayo2.pth')
    
    print("Modelo guardado en 'asl_cnn_final.pth'")

    # Evaluamos en el conjunto de test para ver el rendimiento final
    torch.cuda.empty_cache()
    test_accuracy = evaluate(trained_model, test_loader)
    print(f'Accuracy final en test: {test_accuracy:.2f}%')
