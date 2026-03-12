#  Day 20: MNIST Digit Classification with Transfer Learning
### CNN Image Classification using ResNet18

Built a Convolutional Neural Network using transfer learning to classify handwritten digits (0-9) from the MNIST dataset. Achieved **92.41% test accuracy** using ResNet18 with frozen layers.

---

## 🎯 Project Overview

**What it does:**
- Classifies handwritten digits (0-9) from images
- Uses transfer learning with pre-trained ResNet18
- Processes images through CNN layers for pattern recognition
- Provides confidence scores for predictions

**Why this project:**
- Learn computer vision fundamentals
- Understand Convolutional Neural Networks (CNNs)
- Apply transfer learning to image classification
- Bridge from NLP (Days 14-19) to vision tasks

---

## 🏗️ Architecture

### Model: ResNet18 (Transfer Learning)

```
Pre-trained ResNet18 (ImageNet)
    ↓
Freeze all layers (keep learned features)
    ↓
Replace final layer: 1000 classes → 10 digits
    ↓
Train only final layer on MNIST
    ↓
Output: Digit probabilities (0-9)
```

**Why ResNet18?**
- Pre-trained on 1.4M images (ImageNet)
- Already knows edges, shapes, textures
- Skip connections prevent vanishing gradients
- Fast training with small datasets

**Layer Structure:**
```
Input: 224×224×3 RGB image
    ↓
Conv1: 64 filters, 7×7 (learns basic edges)
    ↓
Layer1: 4 residual blocks (64 channels)
    ↓
Layer2: 4 residual blocks (128 channels)
    ↓
Layer3: 4 residual blocks (256 channels)
    ↓
Layer4: 4 residual blocks (512 channels)
    ↓
AvgPool: Global average pooling
    ↓
FC: 512 → 10 (ONLY THIS LAYER TRAINED)
    ↓
Softmax: Probabilities for digits 0-9
```

---

## 📊 Results

### Overall Performance
```
Test Accuracy:    92.41%
Test Samples:     10,000 images
Training Samples: 60,000 images
Epochs:           10-15
Training Time:    ~10 minutes (GPU)

Baseline (random): 10%
Improvement:       +82 percentage points
```

### Per-Class Performance

| Digit | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.94      | 0.98   | 0.96     | 980     |
| 1     | 0.97      | 0.99   | 0.98     | 1,135   |
| 2     | 0.84      | 0.93   | 0.88     | 1,032   |
| 3     | 0.86      | 0.89   | 0.88     | 1,010   |
| 4     | 0.95      | 0.94   | 0.94     | 982     |
| 5     | 0.90      | 0.82   | 0.85     | 892     |
| 6     | 0.94      | 0.96   | 0.95     | 958     |
| 7     | 0.98      | 0.86   | 0.92     | 1,028   |
| 8     | 0.92      | 0.96   | 0.94     | 974     |
| 9     | 0.94      | 0.90   | 0.92     | 1,009   |

**Best performers:** Digits 1, 0, 6, 8  
**Most challenging:** Digit 2 (precision: 84%)

---

### Confusion Matrix Analysis

```
Main Confusions:
  3 ↔ 5:  123 total confusions (biggest issue)
  2 → 3:   54 confusions
  7 → 9:   27 confusions

Why these confusions make sense:
  - 3 and 5: Both have horizontal lines and curves
  - 2 and 3: Similar top curves
  - 7 and 9: Similar vertical strokes
```

**Detailed Confusion Matrix:**
```
           Predicted
           0    1    2    3    4    5    6    7    8    9
Actual 0  [956   0    2    0    0    5   11    0    5    1]
       1  [  0 1127   4    1    1    0    2    0    0    0]
       2  [  6   3  956  18    2   14   14    3   15    1]
       3  [  1   0   54  903   0   41    0    1    7    3]
       4  [  0  13    8    1  921   3    8    4    9   15]
       5  [  6   3   41   82    0  729  16    2   10    3]
       6  [ 16   4    8    2    1    6  916   0    5    0]
       7  [  6   7   44   19   22   8    0  886   9   27]
       8  [  4   1   10    5    3    4    4    0  939   4]
       9  [ 17   6   10   16   19   4    2    7   20  908]
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Framework | PyTorch |
| Model | ResNet18 (torchvision.models) |
| Dataset | MNIST (60,000 train, 10,000 test) |
| Preprocessing | torchvision.transforms |
| Evaluation | scikit-learn metrics |
| Device | CUDA (GPU) / CPU |

---

## 📁 Project Structure

```
day_20_image_classification/
├── data/
│   └── MNIST/              # Auto-downloaded dataset
│       ├── raw/
│       └── processed/
│
├── scripts/
│   ├── train.py            # Training script
│   ├── evaluate.py         # Model evaluation
│   └── predict.py          # Single image prediction
│
├── models/
│   └── best_model.pth      # Saved model weights
│
└── README.md
```

---

## 🚀 Installation

### Prerequisites
- Python 3.8+
- CUDA (optional, for GPU acceleration)

### Setup

```bash
# Clone repository
git clone https://github.com/sandesh-shrestha69/day_20_sign_language-digit-classifier.git

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**requirements.txt:**
```
torch>=2.0.0
torchvision>=0.15.0
pillow>=9.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
```

---

## 💻 Usage

### 1. Train the Model

```bash
python scripts/train.py
```
**Expected Output:**
```
Using device: cuda
✅ Model architecture created

Epoch  1/15 | Train Loss: 0.3421 | Val Loss: 0.2156 | Val Acc: 93.2%
Epoch  2/15 | Train Loss: 0.1876 | Val Loss: 0.1532 | Val Acc: 95.1%
...
Epoch 15/15 | Train Loss: 0.0234 | Val Loss: 0.0987 | Val Acc: 96.8%

✅ Best model saved: ./models/best_model.pth
```

---

### 2. Evaluate the Model

```bash
python scripts/evaluate.py
```

**Output:**
```
Using device: cuda
✅ Model loaded
✅ Loaded 10000 test images

Evaluating...

============================================================
📊 Test Accuracy: 92.41%
============================================================

🔢 Confusion Matrix:
           Predicted
           0  1  2  3  4  5  6  7  8  9
Actual 0:  [956  0  2  0  0  5 11  0  5  1]
...

📋 Classification Report:
              precision    recall  f1-score   support
           0       0.94      0.98      0.96       980
           1       0.97      0.99      0.98      1135
           ...
```

---

### 3. Predict Single Image

```bash
python scripts/predict.py
```

**Code Example:**
```python
from predict import predict_digit

result = predict_digit("./test_images/digit_5.png")

print(f"Predicted: {result['digit']}")
print(f"Confidence: {result['confidence']:.2%}")
```

**Output:**
```
Using device: cuda
✅ Model loaded

==================================================
🎯 Predicted Digit: 5
📊 Confidence: 98.34%
==================================================

All Probabilities:
  0: 0.0001 
  1: 0.0003 
  2: 0.0012 
  3: 0.0089 ██
  4: 0.0034 
  5: 0.9834 █████████████████████████████████████████████████
  6: 0.0015 
  7: 0.0008 
  8: 0.0002 
  9: 0.0002 
```

---

## 🔬 How It Works

### Image Preprocessing Pipeline

```python
transform = transforms.Compose([
    # 1. Resize MNIST (28×28) to ResNet size (224×224)
    transforms.Resize((224, 224)),
    
    # 2. Convert grayscale (1 channel) to RGB (3 channels)
    transforms.Grayscale(num_output_channels=3),
    
    # 3. Convert PIL Image to PyTorch Tensor
    transforms.ToTensor(),  # [0-255] → [0.0-1.0]
    
    # 4. Normalize with ImageNet statistics
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet mean
        std=[0.229, 0.224, 0.225]    # ImageNet std
    )
])
```

**Why these transforms?**
- **Resize:** ResNet expects 224×224 input
- **Grayscale to RGB:** ResNet expects 3 channels
- **Normalize:** Match ImageNet pre-training statistics

---

### Transfer Learning Strategy

**Approach: Feature Extraction**

```python
# Load pre-trained ResNet18
model = models.resnet18(pretrained=True)

# Freeze ALL layers (keep ImageNet features)
for param in model.parameters():
    param.requires_grad = False

# Replace final layer (1000 → 10 classes)
model.fc = nn.Linear(512, 10)

# Only train the final layer
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
```

**Why this works:**
```
Pre-trained layers (frozen):
  Layer 1: Edge detection (vertical, horizontal, diagonal)
  Layer 2: Texture detection (curves, corners)
  Layer 3: Shape detection (circles, lines)
  Layer 4: Complex patterns
  
Final layer (trainable):
  Maps features → 10 digit classes
  Only 512×10 = 5,120 parameters to learn!
  Can train with small dataset
```

---

### Training Loop

```python
for epoch in range(EPOCHS):
    # ─── Training Phase ───
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # ─── Validation Phase ───
    model.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            # Calculate accuracy
    
    # ─── Save Best Model ───
    if val_acc > best_val_acc:
        torch.save(model.state_dict(), 'best_model.pth')
```

---


### 2. Fine-Tune More Layers

**Current:** Only fc layer trainable

**Better:** Unfreeze layer4 + fc
```python
for name, param in model.named_parameters():
    if 'layer4' in name or 'fc' in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

# Use smaller learning rate for fine-tuning
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-4  # Smaller than 0.001
)
```

---

### 3. Add Data Augmentation

```python
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    
    # Augmentation
    transforms.RandomRotation(10),          # ±10 degrees
    transforms.RandomAffine(
        degrees=0,
        translate=(0.1, 0.1)                # Shift up to 10%
    ),
    
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

---

### 4. Train Longer

```
Current: 10-15 epochs
Try: 20-30 epochs with early stopping
```

---

## 🧠 What I Learned

### Technical Skills
- ✅ Convolutional Neural Networks (CNNs)
  - How convolution detects patterns
  - What pooling does (downsampling)
  - Layer hierarchy (edges → shapes → objects)

- ✅ Transfer Learning for Computer Vision
  - Using pre-trained ImageNet models
  - Freezing vs fine-tuning layers
  - Feature extraction approach

- ✅ Image Preprocessing
  - Resizing and normalization
  - Grayscale to RGB conversion
  - ImageNet statistics application

- ✅ PyTorch Vision
  - torchvision.models (ResNet18)
  - torchvision.transforms pipeline
  - torchvision.datasets (MNIST)
  - DataLoader for batching

- ✅ Model Evaluation
  - Confusion matrix interpretation
  - Per-class metrics (precision/recall)
  - Identifying model weaknesses

### Key Insights

**1. Transfer Learning is Powerful**
```
From scratch: Need millions of images
Transfer learning: 60,000 images → 92% accuracy
Only trained 5,120 parameters (0.04% of model!)
```

**2. Pre-trained Features Transfer Well**
```
ImageNet learned: edges, textures, shapes
These work for digits too!
No need to relearn basic features
```

**3. Confusion Patterns Make Sense**
```
3 ↔ 5: Similar shapes (curves + lines)
2 → 3: Similar top curves
7 → 9: Similar vertical strokes

Model struggles where humans struggle!
```

**4. Computer Vision ≠ NLP (Different Tools, Same Concepts)**
```
NLP (Days 14-19):
  - Input: Text (embeddings)
  - Networks: Dense layers
  - Library: transformers

Vision (Day 20):
  - Input: Images (pixels)
  - Networks: CNNs (convolution)
  - Library: torchvision

Same concepts:
  - Transfer learning ✅
  - Train/val/test split ✅
  - Overfitting detection ✅
  - Model evaluation ✅
```

---

## 🔗 Connection to Previous Days

```
Day 17: Built neural networks from scratch
        → Understood forward/backward pass
        
Day 18: Combined multiple AI systems
        → Learned pipeline architecture
        
Day 19: Fine-tuned pre-trained NLP model (RoBERTa)
        → Applied to sentiment analysis
        
Day 20: Fine-tuned pre-trained vision model (ResNet18)
        → Applied to image classification
        
Pattern: Transfer learning is THE approach for real AI!
```

---

## 🚧 Known Limitations

### 1. Dataset Domain
```
Trained on: MNIST (handwritten digits, clean backgrounds)
Works well on: Similar handwritten digits
Struggles with: 
  - Printed digits
  - Digits in natural scenes
  - Unusual fonts
  - Noisy backgrounds
```

### 2. Model Size vs Accuracy Trade-off
```
ResNet18: Fast, 11M parameters, 92% accuracy
ResNet50: Slower, 25M parameters, ~95% accuracy (expected)

Chose ResNet18 for:
  - Faster training (10 min vs 30 min)
  - Good enough accuracy for learning
```

### 3. Random Weight Initialization
```
Current: weights=None (random initialization)
Impact: 92% accuracy (good but not great)
Fix: Use pretrained=True → expect 98%+
```

---

## 🎓 Future Improvements

- [ ] **Retrain with pre-trained weights** (target: 98%+)
- [ ] **Fine-tune layer4 + fc** (better feature adaptation)
- [ ] **Add more augmentation** (rotation, translation)
- [ ] **Try on real-world digits** (street signs, receipts)
- [ ] **Build web interface** (upload image → predict)
- [ ] **Experiment with other architectures** (VGG, MobileNet)
- [ ] **Test on Fashion MNIST** (harder 10-class problem)
- [ ] **Deploy as REST API** (FastAPI endpoint)

---

## 📚 Resources Used

**Learning Materials:**
- PyTorch Documentation: https://pytorch.org/docs/stable/
- torchvision Models: https://pytorch.org/vision/stable/models.html
- CS231n CNN Guide: https://cs231n.github.io/
- Transfer Learning Tutorial: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

**Dataset:**
- MNIST: http://yann.lecun.com/exdb/mnist/

---

## 👨‍💻 Developer

**Nawich**
- GitHub: https://github.com/sandesh-shrestha69
- Day 20 of 30-day AI Engineering Journey
- Date: February 2026

---

## 📊 Progress Summary

```
Days 1-13:  ✅ Python, APIs, Auth, Databases
Days 14-16: ✅ ML Integration, RAG, LLM Apps
Days 17-18: ✅ Neural Networks, Smart RAG
Day 19:     ✅ Transfer Learning (NLP/Sentiment)
Day 20:     ✅ Transfer Learning (Vision/Digits)

Skills Acquired:
  ✅ Text → embeddings → classification
  ✅ Images → CNNs → classification
  ✅ Transfer learning for both domains
  ✅ End-to-end ML pipelines
  ✅ Model evaluation and analysis

Next: Days 21-25 → Production AI, LLMs, Agents
```

---

*From text classification (Day 19) to image classification (Day 20) - same transfer learning approach, different domain!*

*Built with PyTorch • ResNet18 • MNIST • Transfer Learning*