"""
Módulo para ver datos sobre los modelos guardados.

Requisitos:
    torch.
"""
import torch
if __name__ == '__main__':
    ckeckpoint = torch.load('asl_cnn_checkpoint.pth', weights_only=True)
    history = ckeckpoint['history']
    # Si quieres ver la media por ejemplo
    mean_train_loss = sum(history['train_loss']) / len(history['train_loss'])
    mean_dev_loss = sum(history['dev_loss']) / len(history['dev_loss'])
    mean_train_acc = sum(history['train_acc']) / len(history['train_acc'])
    mean_dev_acc = sum(history['dev_acc']) / len(history['dev_acc'])
    print(f'Mean Train Loss: {mean_train_loss:.4f}')
    print(f'Mean Dev Loss: {mean_dev_loss:.4f}')
    print(f'Mean Train Accuracy: {mean_train_acc:.2f}%')
    print(f'Mean Dev Accuracy: {mean_dev_acc:.2f}%')
