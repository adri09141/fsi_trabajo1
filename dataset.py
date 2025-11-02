import random
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

# --- Parámetros ---
train_dir = "asl_alphabet_train"
test_dir = "asl_alphabet_test"
batch_size = 64
img_size = (128, 128)
val_split = 0.2
# NOTA: use_subset=True y subset_size=10000 significa que solo usas 10k imágenes
# del set de train. Si quieres usarlo todo, pon use_subset=False
use_subset = False 
subset_size = 20000
seed = 42

random.seed(seed)

# --- Transforms ---
train_transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.RandomAffine(
        degrees=0,           # La rotación ya se hizo arriba
        translate=(0.05, 0.05), # Mueve la imagen 5% (simula movimiento)
        scale=(0.95, 1.05),  # Zoom de 95% a 105% (simula cercanía)
        shear=5              # Inclina la imagen 5 grados
    ),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.05, contrast=0.05),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

val_transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

test_transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# --- Cargar datasets base ---
train_base_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
val_base_dataset = datasets.ImageFolder(root=train_dir, transform=val_transform)
test_dataset = datasets.ImageFolder(root=test_dir, transform=test_transform)

base_info = val_base_dataset
class_names = base_info.classes

# --- Subconjunto opcional (sobre el set de training) ---
indices = list(range(len(base_info)))
if use_subset and subset_size < len(indices):
    indices = random.sample(indices, subset_size)

# --- Split entrenamiento/validación (sobre el set de training) ---
random.shuffle(indices)
n_val = int(len(indices) * val_split)
val_indices = indices[:n_val]
train_indices = indices[n_val:]

train_dataset = Subset(train_base_dataset, train_indices)
val_dataset = Subset(val_base_dataset, val_indices)

# --- DataLoaders ---
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

# --- Información ---
def n_classes():
    return len(base_info.classes)

if __name__ == '__main__':
    num_classes = n_classes()
    if use_subset:
        print(f"Usando un subconjunto de {subset_size}")
        print(f"Subconjunto Entrenamiento: {len(train_dataset)}")
        print(f"Subconjunto Validación: {len(val_dataset)}")
    else:
        print(f"Total imágenes de entrenamiento+validación (base): {num_classes}")
    print(f"Total imágenes de Test: {len(test_dataset)}")
    print(f"Clases: {n_classes()}")
    print(f"Nombres de clases: {class_names}")
