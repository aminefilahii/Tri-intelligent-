import json, torch, torch.nn as nn, torch.optim as optim
from torchvision.models import resnet18, ResNet18_Weights
from pathlib import Path
from sklearn.metrics import accuracy_score
from datasets import get_dataloaders

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def train_one_epoch(model, loader, criterion, opt):
    model.train(); total=0; preds=[]; gts=[]
    for x,y in loader:
        x,y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad()
        out = model(x)
        loss = criterion(out,y); loss.backward(); opt.step()
        total += loss.item()*x.size(0)
        preds += out.argmax(1).tolist(); gts += y.tolist()
    return total/len(loader.dataset), accuracy_score(gts,preds)

@torch.no_grad()
def eval_epoch(model, loader, criterion):
    model.eval(); total=0; preds=[]; gts=[]
    for x,y in loader:
        x,y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss = criterion(out,y); total += loss.item()*x.size(0)
        preds += out.argmax(1).tolist(); gts += y.tolist()
    return total/len(loader.dataset), accuracy_score(gts,preds)

def main():
    out = Path("checkpoints"); out.mkdir(exist_ok=True)
    train_loader, val_loader, _, classes = get_dataloaders("data", img_size=224, batch_size=32)

    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best = 0.0
    for epoch in range(15):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, opt)
        va_loss, va_acc = eval_epoch(model, val_loader, criterion)
        print(f"[{epoch:02d}] train loss {tr_loss:.4f} acc {tr_acc:.3f} | val loss {va_loss:.4f} acc {va_acc:.3f}")
        if va_acc > best:
            best = va_acc
            torch.save(model.state_dict(), out/"model_resnet18.pth")
            with open(out/"class_mapping.json","w") as f:
                json.dump({i:c for i,c in enumerate(classes)}, f)
    print("OK ✅ | best val acc:", best)

if __name__ == "__main__":
    main()