import torch
from PIL import Image
from torchvision import transforms

from model import CNN

class_names = [
    "speed_limit",
    "stop",
    "yield",
    "no_entry",
    "pedestrian_crossing",
    "traffic_light",
    "turn_left",
    "turn_right",
    "school_zone",
    "railroad_crossing"
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image_path=r"C:\Users\neelu\PycharmProjects\FastAPIProject1\Road-sign-CNN\data\road_signs\Test\00004.png"

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5])
])

image = Image.open(image_path).convert("RGB")
image = transform(image).unsqueeze(0).to(device)

model = CNN(num_classes=len(class_names)).to(device)
model.load_state_dict(torch.load("models/road-sign-cnn.pth"), map_location=torch.device('cpu') )
model.eval()

with torch.no_grad():
    output = model(image)
    probability = torch.softmax(output, dim=1)
    confidence, predicted_class = torch.max(probability, dim=1)

print("Prediction: ", class_names[predicted_class.item()])
print("Confidence:", round(confidence.item() * 100, 2), "%")
