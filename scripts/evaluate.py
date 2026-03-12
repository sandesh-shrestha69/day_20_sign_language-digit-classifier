from torchvision import models, datasets, transforms
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using device {device}")

def load_model():
    model = models.resnet18(weights=None)

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 10)
    return model.to(device)

test_dataset = datasets.MNIST(
        root= "./data",
        train= False,
        download= True,
        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    )

test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2) 

all_pred = []
all_labels = []

model = load_model()
model.load_state_dict(torch.load('./models/best_model.pth'))
model.eval()

test_correct = 0
test_total = 0
with torch.no_grad():
    for images,labels in test_dataloader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _,predicted = torch.max(outputs.data, 1)
        
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()

        all_pred.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

#calculate accuracy
test_acc = 100 * test_correct/test_total
print(f"Test accuracy: {test_acc:.2f}%")
#confusion matrix
cm = confusion_matrix(all_labels, all_pred)

print("Confusion Matrix")
print("     predicted")
print("     0  1  2  3  4  5  6  7  8  9")
for i in range(10):
    print(f"{i}: {cm[i]}")


#classification report
print("\n Classification report")
print(classification_report(
    all_labels,
    all_pred,
    target_names=[str(i) for i in range(10)]
))

print("Evaluation completed")