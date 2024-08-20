import os
import shutil
from tqdm import tqdm
import torchvision.transforms as transforms
import torchvision.datasets as datasets

def prepare_data(data_root, train_dir, test_dir, val_dir, class_names):
    for dir_name in [train_dir, test_dir, val_dir]:
        for class_name in class_names:
            os.makedirs(os.path.join(dir_name, class_name), exist_ok=True)

    for class_name in class_names:
        source_dir_train = os.path.join(data_root, 'train', class_name)
        for i, file_name in enumerate(tqdm(os.listdir(source_dir_train))):
            if i % 10 != 0:
                dest_dir = os.path.join(train_dir, class_name)
            else:
                dest_dir = os.path.join(val_dir, class_name)
            shutil.copy(os.path.join(source_dir_train, file_name), os.path.join(dest_dir, file_name))

    for class_name in class_names:
        source_dir_test = os.path.join(data_root, 'test', class_name)
        for i, file_name in enumerate(tqdm(os.listdir(source_dir_test))):
            dest_dir = os.path.join(test_dir, class_name)
            shutil.copy(os.path.join(source_dir_test, file_name), os.path.join(dest_dir, file_name))

def get_dataloaders(train_dir, val_dir, test_dir, batch_size):
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(train_dir, train_transforms)
    val_dataset = datasets.ImageFolder(val_dir, val_transforms)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=2)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_dataset = datasets.ImageFolder(test_dir, test_transforms)

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_dataloader, val_dataloader, test_dataloader
