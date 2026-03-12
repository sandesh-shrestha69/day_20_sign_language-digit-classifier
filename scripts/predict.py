"""
Predict digit from single image
"""
from torchvision import models, transforms
import torch
import torch.nn as nn  # ✅ Add this import
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

def predict_digit(image_path):
    # Load Model ───
    print("Loading model...")
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(512, 10)  
    model.load_state_dict(torch.load('./models/best_model.pth'))  
    model = model.to(device)  
    model.eval()              
    
    # Load Image 
    print(f"Loading image: {image_path}...")
    image = Image.open(image_path).convert('RGB')  
    
    #  Transform 
    print("Transforming...")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    image_tensor = transform(image).unsqueeze(0) 
    image_tensor = image_tensor.to(device)
    
    # Predict
    print("Predicting...")
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    return {
        "digit": predicted_class,
        "confidence": confidence,
        "all_probabilities": probabilities[0].cpu().numpy()
    }

if __name__ == '__main__':
    result = predict_digit("./image.png")  
    
    print(f"\n{'='*50}")
    print(f"Predicted Digit: {result['digit']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"{'='*50}")
    
    print(f"\nAll Probabilities:")
    for digit, prob in enumerate(result['all_probabilities']):
        bar = '█' * int(prob * 50)
        print(f"  {digit}: {prob:.4f} {bar}")
    print()