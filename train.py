
import torch
import torch.nn.functional as F
# --- ¡Importante! Importar desde tus otros archivos ---
from dataset import train_loader, val_loader, n_classes, test_loader
from model import model, criterion, optimizer # Importa el modelo, criterio y optimizador instanciados
from showGraph import plot_training_history

# (Ligera mejora) Hago que evaluate también devuelva el accuracy para poder usarlo en el bucle
def evaluate(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()  # Poner el modelo en modo evaluación
    correct = 0
    total = 0
    with torch.no_grad():  # No calcular gradientes durante la evaluación
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # Mover datos al dispositivo
            outputs = model(inputs)  # Forward pass
            _, predicted = torch.max(outputs.data, 1) # Usar .data para obtener solo los valores
            total += labels.size(0) # Contar el total de muestras en el batch
            correct += (predicted == labels).sum().item()  # Actualizar el contador de aciertos
    accuracy = 100 * correct / total if total > 0 else 0.0
    print(f'Evaluación - Total: {total}, Correctos: {correct}') # Para depurar si es necesario
    return accuracy


def train_with_validation(model, train_loader, dev_loader, criterion, optimizer, epochs=5):
    """
    Entrena 'model' durante 'epochs' usando train_loader y evalúa en dev_loader.
    Imprime acc y loss de train y dev por época.
    Devuelve (modelo_entrenado, history) donde history contiene listas por época.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Entrenando en dispositivo: {device}")
    model.to(device) # Asegurarse de que el modelo está en el dispositivo correcto

    history = {
        'train_loss': [],
        'train_acc': [],
        'dev_loss': [],
        'dev_acc': [],
    }

    num_classes = n_classes() # Obtenemos el número de clases

    for epoch in range(epochs):
        model.train() # Poner el modelo en modo entrenamiento
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for i, (inputs, labels) in enumerate(train_loader): # Mejor nombrar explícitamente i
            optimizer.zero_grad()  # Limpiar gradientes
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)  # Forward pass

            # --- CORRECCIÓN: CrossEntropyLoss espera índices, no one-hot ---
            loss = criterion(outputs, labels) # Pasar directamente los índices de clase
            # --- FIN CORRECCIÓN ---

            loss.backward()  # Backward pass
            optimizer.step()  # Actualizar pesos

            running_loss += loss.item()

            # Calcular accuracy en el batch de entrenamiento
            _, predicted_train = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted_train == labels).sum().item()

            # Imprimir progreso cada N batches (opcional)
            if (i + 1) % 50 == 0: # Imprime cada 50 batches
                 print(f'  [Época {epoch + 1}, Batch {i + 1:4d}] Pérdida ent.: {loss.item():.4f}')


        avg_train_loss = running_loss / len(train_loader)
        train_acc = 100.0 * correct_train / total_train if total_train > 0 else 0.0

        # ---- Validación (loss y accuracy) ----
        model.eval() # Poner el modelo en modo evaluación
        dev_running_loss = 0.0
        correct_dev = 0
        total_dev = 0
        with torch.no_grad(): # Desactivar cálculo de gradientes para validación
            for inputs, labels in dev_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)

                # --- CORRECCIÓN: CrossEntropyLoss espera índices, no one-hot ---
                dev_loss = criterion(outputs, labels)
                # --- FIN CORRECCIÓN ---
                dev_running_loss += dev_loss.item()

                # Calcular accuracy en el batch de validación
                _, predicted_dev = torch.max(outputs.data, 1)
                total_dev += labels.size(0)
                correct_dev += (predicted_dev == labels).sum().item()

        avg_dev_loss = dev_running_loss / len(dev_loader)
        dev_acc = 100.0 * correct_dev / total_dev if total_dev > 0 else 0.0
        # También puedes usar tu función evaluate aquí si prefieres: dev_acc = evaluate(model, dev_loader)


        # Guardar histórico
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        history['dev_loss'].append(avg_dev_loss)
        history['dev_acc'].append(dev_acc)

        # Log por época
        print(f'\n--- Época {epoch + 1}/{epochs} Finalizada ---')
        print(f'Pérdida Entrenamiento : {avg_train_loss:.4f} | Acc Entrenamiento : {train_acc:.2f}%')
        print(f'Pérdida Validación   : {avg_dev_loss:.4f} | Acc Validación   : {dev_acc:.2f}%\n')

    print('--- Entrenamiento Completado ---')
    return model, history

# --- ¡Aquí es donde inicias el entrenamiento! ---
if __name__ == '__main__':
    num_epochs_to_train = 20
    print(f'\n--- Época {0}/{num_epochs_to_train} Finalizada ---')
    trained_model, training_history = train_with_validation(
        model=model,
        train_loader=train_loader,
        dev_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        epochs=num_epochs_to_train
    )
    plot_training_history(training_history)
    # Cuidado no esta termino porque podria ser que la epoca 25 sea la mejor y la 30 no (ESTO GUARDARIA LA ULTIMA EPOCA)
    print("Guardando el modelo entrenado...")
    torch.save(trained_model.state_dict(), 'asl_cnn_final.pth')
    print("Modelo guardado en 'asl_cnn_final.pth'")

    test_accuracy = evaluate(trained_model, test_loader)  # aquí sí usamos test_loader
    print(f'Accuracy final en test: {test_accuracy:.2f}%')
