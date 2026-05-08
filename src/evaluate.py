import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

from model import CNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

test_dataset = datasets.ImageFolder(root=r'C:\Users\neelu\PycharmProjects\FastAPIProject1\Road-sign-CNN\data\road_signs\Test', transform=transform)

test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

num_classes = len(test_dataset)

model = CNN(num_classes=num_classes).to(device)

model.load_state_dict(torch.load('./checkpoints/road_signs_cnn.pth'), map_location=device)
model.eval()

all_labels = []
all_preds = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())
print("\nClassification Report")
print("---------------------")
print(classification_report(
    all_labels,
    all_preds,
    target_names=test_dataset.classes
))

cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=test_dataset.classes)
disp.plot(xticks_rotation=45)
plt.title("Confusion Matrix")
plt.savefig("outputs/confusion_matrix.png")
plt.show()
