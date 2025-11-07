import torch
import torch.nn.functional as F
# --- Importaciones desde tus otros archivos ---
from dataset import train_loader, val_loader, n_classes, test_loader
from model import model, criterion, optimizer  # Traemos el modelo, la función de pérdida y el optimizador ya definidos
from showGraph import plot_training_history
from torch.optim.lr_scheduler import ReduceLROnPlateau

# -------------------------------------------------------------------
# Función para evaluar el modelo en un conjunto de datos (por ejemplo, test o validación)
# -------------------------------------------------------------------
def evaluate(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()  # Se pone el modelo en modo evaluación (desactiva dropout y batchnorm)
    correct = 0
    total = 0
    with torch.no_grad():  # No necesitamos calcular gradientes al evaluar
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)  # Toma la clase con mayor probabilidad
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total if total > 0 else 0.0
    print(f'Evaluación - Total: {total}, Correctos: {correct}')  # Info útil para revisar
    return accuracy


# -------------------------------------------------------------------
# Función principal de entrenamiento con validación en cada época
# -------------------------------------------------------------------
def train_with_validation(model, train_loader, dev_loader, criterion, optimizer, epochs=5):
    """
    Entrena el modelo durante 'epochs' épocas usando el conjunto de entrenamiento,
    y evalúa su desempeño en validación al final de cada una.
    Devuelve el modelo entrenado y un registro de las métricas.
    """
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',       # Queremos minimizar la pérdida
        factor=0.5,       # Reduce el LR a la mitad
        patience=2       # Espera 2 épocas sin mejora
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

    num_classes = n_classes()  # Obtenemos la cantidad de clases

    for epoch in range(epochs):
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

        # Mostramos resumen de la época
        print(f'\n--- Época {epoch + 1}/{epochs} completada ---')
        print(f'Pérdida Entrenamiento : {avg_train_loss:.4f} | Exactitud Entrenamiento : {train_acc:.2f}%')
        print(f'Pérdida Validación    : {avg_dev_loss:.4f} | Exactitud Validación    : {dev_acc:.2f}%\n')
        scheduler.step(avg_dev_loss)

        # (Opcional) Mostrar el LR actual
        current_lr =scheduler.get_last_lr()[0]
        print(f"Learning rate actual: {current_lr:.6f}\n")


    print('--- Entrenamiento finalizado ---')
    return model, history


# -------------------------------------------------------------------
# Bloque principal — aquí arranca todo
# -------------------------------------------------------------------
if __name__ == '__main__':
    num_epochs_to_train = 5
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

    # Importante: esto guarda el modelo tal como terminó.
    # Si querés guardar el "mejor" modelo (por validación), habría que agregar early stopping.
    print("Guardando el modelo entrenado...")
    torch.save(trained_model.state_dict(), 'asl_cnn_final.pth')
    print("Modelo guardado en 'asl_cnn_final.pth'")

    # Evaluamos en el conjunto de test para ver el rendimiento final
    torch.cuda.empty_cache()
    test_accuracy = evaluate(trained_model, test_loader)
    print(f'Accuracy final en test: {test_accuracy:.2f}%')
