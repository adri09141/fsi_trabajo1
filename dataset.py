import random
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

# --- Parámetros ---
data_dir = "asl_alphabet_train"
batch_size = 64
img_size = (128, 128)
val_split = 0.2
use_subset = True
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

# --- Para saber cuántas imágenes hay y las etiquetas ---
base = datasets.ImageFolder(root=data_dir)  # sin transform, solo para info
indices = list(range(len(base)))

# --- Opcional: tomar un subset aleatorio (representativo si no quieres todo) ---
if use_subset and subset_size < len(indices):
    indices = random.sample(indices, subset_size)

# --- Split simple (no estratificado para mantener simple) ---
random.shuffle(indices)
n_val = int(len(indices) * val_split)
val_indices = indices[:n_val]
train_indices = indices[n_val:]

# --- Crear datasets con transform apropiado (cada ImageFolder aplica su propio transform) ---
dataset_train = datasets.ImageFolder(root=data_dir, transform=train_transform)
dataset_val   = datasets.ImageFolder(root=data_dir, transform=val_transform)

train_dataset = Subset(dataset_train, train_indices)
val_dataset   = Subset(dataset_val, val_indices)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

# --- Información ---
def n_classes():
    return len(base.classes)

if __name__ == '__main__':
    print(f"Total imágenes cargadas: {len(base)}")
    print(f"Entrenamiento: {len(train_dataset)}")
    print(f"Validación: {len(val_dataset)}")
    print(f"Clases: {n_classes()}")
    #pass  # ← opcional, solo para dejar claro que no hace nada
