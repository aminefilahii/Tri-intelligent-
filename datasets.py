from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import json

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def get_class_mapping(mapping_path="class_mapping.json"):
    with open(mapping_path,"r",encoding="utf-8") as f:
        j = json.load(f)
    res = []
    for i in range(len(j)):
        key = str(i) if str(i) in j else i
        res.append(j[key])
    return res

def get_dataloaders(data_dir="data", img_size=224, batch_size=32, num_workers=2, aug=True):
    train_tf = [
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    if aug:
        train_tf.insert(1, transforms.RandomHorizontalFlip())
        train_tf.insert(1, transforms.ColorJitter(0.1,0.1,0.1,0.05))

    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    train_ds = datasets.ImageFolder(f"{data_dir}/train", transform=transforms.Compose(train_tf))
    val_ds   = datasets.ImageFolder(f"{data_dir}/val",   transform=eval_tf)
    test_ds  = datasets.ImageFolder(f"{data_dir}/test",  transform=eval_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, train_ds.classes