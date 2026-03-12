"""download train,test, validate dataset from mnist
augmentate the train  dataset
load pretrained model
freeze all layer
replace fc layer with 10 digits
optimizer the fc parameters with crossentropyloss
start the training loop
validate
save best model in models"""
from torchvision import datasets, models, transforms
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

def train():
    #setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using deveice {device}")

    #load MNIST dataset and apply augmentation
    train_dataset = datasets.MNIST(
        root= "./data",
        train= True,
        download= True,
        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2,contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    )

    val_dataset = datasets.MNIST(
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
    #load daataset
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

    #load model
    model = models.resnet18(pretrained = True)

    #freeze freeze all layer
    for params in model.parameters():
        params.requires_grad = False

    #replace last layer
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 10)
    model = model.to(device)

    #loss
    criterion = nn.CrossEntropyLoss()

    #optimizer
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

    #training loop
    num_of_epoch = 10
    best_val_acc = 0

    for epoch in range(num_of_epoch):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            #forward pass
            outputs =model(images)
            loss = criterion(outputs, labels)

            #backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #statistic
            train_loss = loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_acc = 100 * train_correct/train_total
        avg_train_loss = train_loss / len(train_loader)

        #validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad(): 
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss +=loss.item()
                _,predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct/val_total
        avg_val_loss = val_loss/len(val_loader)

        print(f"Epoch {epoch+1}/{num_of_epoch}")
        print(f"  Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {avg_val_loss:.4f} | Val Acc:   {val_acc:.2f}%")

        #save best model
        if val_acc < best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), './models/best_model.pth')
            
            print(f" saved (best: {best_val_acc:.2f})%")
        print()
    print(f"\n🎉 Training complete! Best validation accuracy: {best_val_acc:.2f}%")

if __name__ == "__main__":
    train()