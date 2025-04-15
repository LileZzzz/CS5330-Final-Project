# This class creates a FastAPI service
import io
from contextlib import asynccontextmanager

import torch
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from torchvision import transforms
from torchvision.models import vit_b_16

# Global variables
model = None
device = None
classes = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]


# Define the model
def load_vit(num_classes=10):
    # Load the pre-trained ViT model
    model = vit_b_16(weights="DEFAULT")
    for param in model.parameters():
        param.requires_grad = False

    # replace the classifier head
    num_features = model.heads.head.in_features
    model.heads.head = torch.nn.Linear(num_features, num_classes)

    return model


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, device

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.mps.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    try:
        # Load model
        model = load_vit()

        # Load fine-tuned weights
        model.load_state_dict(
            torch.load("api/vit_eurosat_best_model.pth", map_location=device)
        )
        model.to(device)
        model.eval()

        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None

    yield

    print("Shut down")


# Create app
app = FastAPI(title="EuroSAT Classification API", lifespan=lifespan)

# ImageNet mean and std for normalization
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

# Image transformation
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ]
)


@app.get("/")
async def root():
    return {
        "message": "EuroSAT Classification API is running",
        "model_status": "loaded" if model is not None else "not loaded",
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Check if model is loaded
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Check file type
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid file format. Please upload a JPEG or PNG image.",
        )

    try:
        # Read image file
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))

        # Convert RGBA to RGB
        if image.mode == "RGBA":
            image = image.convert("RGB")

        # Preprocess image
        image_tensor = transform(image).unsqueeze(0).to(device)

        # Get prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]

        # Get top 3 results
        top_probs, top_classes = torch.topk(probabilities, 3)

        # Append results
        results = []
        for i in range(3):
            idx = top_classes[i].item()
            prob = top_probs[i].item() * 100
            results.append({"class": classes[idx], "probability": f"{prob:.2f}%"})

        return JSONResponse(content={"predictions": results})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=1001)
