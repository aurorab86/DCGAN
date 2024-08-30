import torch
import torch.utils.data
import torchvision
import torchvision.datasets as dataset
import torchvision.transforms as transforms

def load_dataset(PATH, img_size, batch_size):
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    img_dataset = dataset.ImageFolder(root=PATH,
                            transform=transform)

    data_loader = torch.utils.data.DataLoader(dataset=img_dataset,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            drop_last=True)
    
    return data_loader