# 🚀 PCB DEFECT DETECTION - TRAINING GUIDE

**Hướng dẫn chi tiết để train model trên server GPU RTX A4000/3090**

## 📋 Prerequisites

### System Requirements
- **GPU**: RTX A4000 (16GB) hoặc RTX 3090 (24GB)
- **RAM**: 16GB+ khuyến nghị
- **Storage**: 20GB+ free space
- **OS**: Windows 10/11, Ubuntu 18.04+
- **Python**: 3.8+

### Dataset Structure Required
```
pcb-defect-dataset/
├── train/
│   ├── images/     # Training images (.jpg)
│   └── labels/     # YOLO format labels (.txt)
├── val/
│   ├── images/     # Validation images
│   └── labels/     # YOLO format labels
└── test/
    ├── images/     # Test images
    └── labels/     # YOLO format labels
```

---

## 🔧 SETUP COMMANDS (Copy & Run)

### 1. Clone Repository
```bash
git clone https://github.com/theanh7/PCB-MODEL.git
cd PCB-MODEL
```

### 2. Install Dependencies
```bash
# Install tất cả dependencies
pip install -r requirements.txt

# Kiểm tra CUDA
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

### 3. Verify Dataset
```bash
# Kiểm tra dataset structure
ls -la pcb-defect-dataset/
ls -la pcb-defect-dataset/train/images/ | head -5
ls -la pcb-defect-dataset/train/labels/ | head -5
```

---

## 🧪 TESTING PHASE (Highly Recommended)

### Step 1: Test System & GPU Optimization
```bash
# Test comprehensive - CHẠY LỆNH NÀY TRƯỚC!
python test_optimized_training.py
```

**Expected Output:**
```
✅ GPU: RTX 3090/A4000
✅ VRAM: 24.0GB/16.0GB
✅ Memory efficiency: 60-80%
✅ SẴN SÀNG CHO PRODUCTION TRAINING!
```

### Step 2: Quick Validation (Optional)
```bash
# Test nhanh 2 epochs
python test_enhanced_training.py
```

**Expected Output:**
```
SUCCESS: Model is LEARNING! (Non-zero coord/class loss)
```

---

## 🚀 PRODUCTION TRAINING

### Full Training Command
```bash
# CHẠY LỆNH CHÍNH - OPTIMIZED CHO RTX A4000/3090
python train.py
```

### Training Configuration (Auto-optimized)
```
✅ Batch Size: 32 (tối ưu cho 16-24GB VRAM)
✅ Num Workers: 12 (multi-core CPU)
✅ Mixed Precision: Enabled (tiết kiệm 40% VRAM)
✅ Pin Memory: Enabled (tăng tốc transfer)
✅ Epochs: 100 (early stopping enabled)
```

### Monitor Training Progress
```bash
# Training sẽ tự động hiển thị:
# - Loss values (Coord, Conf, Class)
# - Learning rate
# - GPU memory usage
# - Training time per epoch
# - Best model checkpoints

# Ví dụ output:
# Epoch 1/100: Loss=45.23, Coord=4.77, Conf=17.42, Class=4.20, LR=0.001000
# ✅ New best model saved: outputs/best_model.pth
```

---

## 📊 EVALUATION & TESTING

### Evaluate Trained Model
```bash
# Đánh giá model sau khi train xong
python evaluate.py
```

### Test Single Image Inference
```bash
# Test inference trên 1 ảnh
python inference.py --model outputs/best_model.pth --input path/to/test_image.jpg
```

---

## 📈 PERFORMANCE EXPECTATIONS

### Training Speed (RTX 3090)
- **Images/second**: ~200-400
- **Time per epoch**: ~30-60 seconds (tùy dataset size)
- **Full training (100 epochs)**: ~1-2 hours
- **Memory usage**: ~10-12GB / 24GB

### Training Speed (RTX A4000)
- **Images/second**: ~150-300
- **Time per epoch**: ~45-90 seconds
- **Full training (100 epochs)**: ~1.5-2.5 hours  
- **Memory usage**: ~10-12GB / 16GB

---

## 🔍 TROUBLESHOOTING

### Common Issues & Solutions

#### 1. CUDA Out of Memory
```bash
# Giảm batch size nếu OOM
# Edit train.py line 322:
'batch_size': 24,  # Thay vì 32
```

#### 2. Dataset Not Found
```bash
# Kiểm tra đường dẫn dataset
ls -la pcb-defect-dataset/

# Nếu không có, tạo symbolic link:
ln -s /path/to/your/dataset pcb-defect-dataset
```

#### 3. Slow Data Loading
```bash
# Giảm num_workers nếu CPU yếu
# Edit train.py line 323:
'num_workers': 8,  # Thay vì 12
```

#### 4. Model Not Learning (Zero Loss)
```bash
# Kiểm tra dataset labels
python -c "
import os
labels_dir = 'pcb-defect-dataset/train/labels'
labels = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
print(f'Found {len(labels)} label files')
if labels:
    with open(os.path.join(labels_dir, labels[0])) as f:
        print(f'Sample label: {f.read().strip()}')
"
```

---

## 💾 OUTPUT FILES

### Training Outputs (Tự động tạo)
```
outputs/
├── best_model.pth              # Best model checkpoint
├── final_model.pth             # Final model  
├── training_20250127_143022.log # Training log
└── training_curves.png         # Loss curves visualization
```

### Model Checkpoints
- **best_model.pth**: Model với validation loss thấp nhất
- **final_model.pth**: Model cuối cùng sau 100 epochs
- **Training log**: Chi tiết quá trình training
- **Loss curves**: Biểu đồ loss để phân tích

---

## 🎯 QUICK START CHECKLIST

```bash
# 1. ✅ Clone repo
git clone https://github.com/theanh7/PCB-MODEL.git && cd PCB-MODEL

# 2. ✅ Install dependencies  
pip install -r requirements.txt

# 3. ✅ Test system (MANDATORY)
python test_optimized_training.py

# 4. ✅ Start training
python train.py

# 5. ✅ Evaluate results
python evaluate.py
```

---

## 📞 SUPPORT

### Nếu gặp vấn đề:

1. **Check GPU status**: `nvidia-smi`
2. **Check CUDA**: `python -c "import torch; print(torch.cuda.is_available())"`
3. **Check logs**: `cat outputs/training_*.log`
4. **Memory usage**: Monitor `nvidia-smi` trong khi training

### Expected Training Success Indicators:
```
✅ Coord Loss > 0 (bbox regression working)
✅ Class Loss > 0 (classification working)  
✅ Total Loss decreasing over epochs
✅ GPU memory usage stable 60-80%
✅ No CUDA errors
```

---

## 🏁 FINAL NOTES

- **Training time**: 1-3 hours depending on dataset size
- **GPU utilization**: Should be >90% during training
- **Memory efficiency**: 60-80% VRAM usage is optimal
- **Early stopping**: Training stops automatically if no improvement
- **Backup**: Model checkpoints saved automatically

**🚀 Ready for production training on RTX A4000/3090!**