from fastapi import FastAPI, File, UploadFile
# from fastapi.staticfiles import StaticFiles
# from fastapi.middleware.cors import CORSMiddleware
import torch
import torchvision.transforms as transforms
import timm 
from PIL import Image 
import io
from imagenet_test_classes import class_labels

app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:3000"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"]
# )
# app.mount("/", StaticFiles(directory="D:/Project/Web Codes/vit-image-classifier/build", html=True), name="static")
#pretrained ViTModel
device = torch.device("cpu")

model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=160)

model.load_state_dict(torch.load('D:/Project/ViT_Model/vit_imagenet160classes.pth', map_location=device))

model.to(device)

model.eval()

#image transforms
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

])


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read()))
    image = transform(image).unsqueeze(0).to(device)


    with torch.no_grad():
        output = model(image)
        predicted_class = torch.argmax(output, dim=1).item()

    predicted_label = class_labels[predicted_class]
    if predicted_class >= len(class_labels):
        return {"error": "Invalid class index", "index":predicted_class}
    return {"filename": file.filename, "prediction": predicted_label}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


# print(outpu) 