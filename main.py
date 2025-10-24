from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

data_dir = "asl_alphabet_train"
batch_size = 12

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])

dataset = datasets.ImageFolder(root=data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

print(len(dataset.classes))
print(dataset.classes)