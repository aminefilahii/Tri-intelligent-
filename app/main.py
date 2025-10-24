import io, json, torch, torch.nn as nn
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from torchvision import transforms
from torchvision.models import resnet18

IMAGENET_MEAN=[0.485,0.456,0.406]; IMAGENET_STD=[0.229,0.224,0.225]
tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

app = FastAPI(title="Tri-intelligent API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

with open("checkpoints/class_mapping.json","r",encoding="utf-8") as f:
    idx2class = {int(k):v for k,v in json.load(f).items()}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(idx2class))
model.load_state_dict(torch.load("checkpoints/model_resnet18.pth", map_location=DEVICE))
model.eval().to(DEVICE)

BIN_RULES = {
    "verre": "Verre", "glass":"Verre",
    "plastic":"Plastique","plastique":"Plastique",
    "metal":"Métal","can":"Métal","aluminum":"Métal",
    "paper":"Emballages & papiers","cardboard":"Emballages & papiers",
}
BIN_COLORS = {"Verre":"#2ecc71","Plastique":"#f39c12","Métal":"#bdc3c7","Emballages & papiers":"#3498db","Non recyclable":"#7f8c8d"}

def to_bin(label:str)->str:
    l = label.lower()
    for k,v in BIN_RULES.items():
        if k in l: return v
    return "Non recyclable"

@app.get("/")
async def root():
    return {"status":"ok","message":"Tri-intelligent API en ligne"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    x = tf(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(x)
        prob = torch.softmax(logits, dim=1).max().item()
        idx  = logits.argmax(1).item()
    label = idx2class[idx]
    bin_name = to_bin(label)
    return {"label":label,"proba":round(prob,4),"bin":bin_name,"bin_color":BIN_COLORS[bin_name]}