import random
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

# ----------------------------
# Parámetros generales
# ----------------------------
train_dir = "asl_alphabet_train"
batch_size = 128
img_size = (128, 128)
val_split = 0.2
test_split = 0.1
seed = 42

random.seed(seed)
torch.manual_seed(seed)

# ----------------------------
# Transformaciones de imagen
# ----------------------------

# Aumentos para entrenamiento
train_transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.RandomCrop(img_size, padding=8),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(
        brightness=0.1,
        contrast=0.1,
        saturation=0.05
    ),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Validación y test → sin aumentos
val_transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


# ----------------------------
# Cargar datasets
# ----------------------------
train_base_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
val_base_dataset = datasets.ImageFolder(root=train_dir, transform=val_transform)

# Guardar nombres de clases
base_info = val_base_dataset
class_names = base_info.classes

# ----------------------------
# División entrenamiento / validación / test
# ----------------------------
indices = list(range(len(base_info)))
random.shuffle(indices)

n_val = int(len(indices) * val_split)   # ej. 20%
n_test = int(len(indices) * test_split) # ej. 10%


val_indices = indices[:n_val]
test_indices = indices[n_val : n_val + n_test]
train_indices = indices[n_val + n_test :]


# Subsets de PyTorch
train_dataset = Subset(train_base_dataset, train_indices)
val_dataset = Subset(val_base_dataset, val_indices) 
test_dataset = Subset(val_base_dataset, test_indices)

# ----------------------------
# DataLoaders
# ----------------------------

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

# ----------------------------
# Obtener número de clases
# ----------------------------
def n_classes():
    return len(base_info.classes)

# ----------------------------
# Información del dataset
# ----------------------------
if __name__ == '__main__':
    num_classes = n_classes()
    print(f"Total imágenes de entrenamiento: {len(train_dataset)}")
    print(f"Total imágenes de validación: {len(val_dataset)}")
    print(f"Total imágenes de test (del set de train): {len(test_dataset)}") # <-- Cambiado
    print(f"Número de clases: {num_classes}")
    print(f"Nombres de clases: {class_names}")
