import random
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, random_split

# --- Parámetros ---
data_dir = "asl_alphabet_train"
batch_size = 64
img_size = (128, 128)
val_split = 0.2
use_subset = False
subset_size = 20000
seed = 42

random.seed(seed)

# --- Transforms ---
train_transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.RandomRotation(5),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

val_transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

# --- CORRECCIÓN: Cargar datasets base (uno para cada transform) ---
# Creamos dos instancias. Es rápido, solo leen la estructura de carpetas.

# train_base_dataset aplicará 'train_transform' cuando se acceda a él.
train_base_dataset = datasets.ImageFolder(root=data_dir, transform=train_transform)
# val_base_dataset aplicará 'val_transform' cuando se acceda a él.
val_base_dataset = datasets.ImageFolder(root=data_dir, transform=val_transform)

# Usamos una de las instancias (son idénticas antes del transform) para la info
# (Podríamos usar train_base_dataset o val_base_dataset, da igual)
_base_info_source = val_base_dataset 

# --- Subconjunto opcional ---
indices = list(range(len(_base_info_source)))
if use_subset and subset_size < len(indices):
    indices = random.sample(indices, subset_size)

# --- Split ---
random.shuffle(indices)
n_val = int(len(indices) * val_split)
val_indices = indices[:n_val]
train_indices = indices[n_val:]
# --- Subsets ---
# El subset de train usará el dataset base CON train_transform
train_dataset = Subset(train_base_dataset, train_indices)
# El subset de val usará el dataset base CON val_transform
val_dataset   = Subset(val_base_dataset, val_indices)

# --- DataLoaders ---
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

# --- Información ---
def n_classes():
    # Usamos la variable de referencia que creamos
    return len(_base_info_source.classes)

if __name__ == '__main__':
    print(f"Total imágenes (según una instancia): {len(_base_info_source)}")
    print(f"Entrenamiento: {len(train_dataset)}")
    print(f"Validación: {len(val_dataset)}")
    print(f"Clases: {n_classes()}")
