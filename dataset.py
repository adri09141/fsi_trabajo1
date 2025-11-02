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
    transforms.RandomHorizontalFlip(p=0.1),
    transforms.RandomVerticalFlip(p=0.1),
    transforms.RandomRotation(5),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
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

_base_info_source = val_base_dataset
# --- ¡AÑADIR ESTO! ---
# Exportamos los nombres de las clases (A, B, C...)
class_names = _base_info_source.classes
# --- FIN DE LA ADICIÓN ---

# --- Subconjunto opcional (sobre el set de training) ---
indices = list(range(len(_base_info_source)))
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
    return len(_base_info_source.classes)

if __name__ == '__main__':
    if use_subset:
        print(f"Usando un subconjunto de {subset_size}")
        print(f"Subconjunto Entrenamiento: {len(train_dataset)}")
        print(f"Subconjunto Validación: {len(val_dataset)}")
    else:
        print(f"Total imágenes de entrenamiento+validación (base): {len(_base_info_source)}")
    print(f"Total imágenes de Test: {len(test_dataset)}")
    print(f"Clases: {n_classes()}")
    print(f"Nombres de clases: {class_names}")
