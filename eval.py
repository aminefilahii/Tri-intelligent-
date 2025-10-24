import json, torch, torch.nn as nn
from torchvision.models import resnet18
from datasets import get_dataloaders
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@torch.no_grad()
def run(model, loader):
    model.eval(); y_true=[]; y_pred=[]
    for x,y in loader:
        x = x.to(DEVICE)
        y_true += y.tolist()
        y_pred += model(x).argmax(1).cpu().tolist()
    return np.array(y_true), np.array(y_pred)

def main():
    _, _, test_loader, classes = get_dataloaders("data", img_size=224, batch_size=32)
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    state = torch.load("checkpoints/model_resnet18.pth", map_location=DEVICE)
    model.load_state_dict(state); model.to(DEVICE)

    y_true, y_pred = run(model, test_loader)
    print(classification_report(y_true, y_pred, target_names=classes, digits=3))
    print("Confusion:\n", confusion_matrix(y_true, y_pred))

if __name__ == "__main__":
    main()