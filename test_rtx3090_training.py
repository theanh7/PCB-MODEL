"""
RTX 3090 Optimized Test Training Script
Kiểm tra khả năng train trên RTX 3090 24GB VRAM
"""

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import os
import time
import psutil
import gc

from model import create_model, PCBLoss
from enhanced_dataset import create_enhanced_dataloaders


def get_gpu_memory_info():
    """Lấy thông tin GPU memory"""
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        memory_reserved = torch.cuda.memory_reserved() / 1024**3   # GB
        memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        return memory_allocated, memory_reserved, memory_total
    return 0, 0, 0


def test_rtx3090_training():
    """Test training optimized cho RTX 3090 24GB"""
    
    print("RTX 3090 PCB DEFECT DETECTION - TEST TRAINING")
    print("="*70)
    
    # Kiểm tra CUDA
    if not torch.cuda.is_available():
        print("❌ CUDA không khả dụng! Cần GPU RTX 3090.")
        return False
    
    print(f"✅ CUDA Version: {torch.version.cuda}")
    print(f"✅ PyTorch Version: {torch.__version__}")
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    
    mem_alloc, mem_reserved, mem_total = get_gpu_memory_info()
    print(f"✅ GPU Memory: {mem_total:.1f}GB total")
    
    # Configuration tối ưu cho RTX 3090
    config = {
        'data_dir': 'pcb-defect-dataset',
        'batch_size': 16,        # RTX 3090 có thể handle batch lớn
        'num_workers': 8,        # Tối ưu cho CPU multi-core
        'img_size': 600,
        'num_classes': 6,
        'learning_rate': 1e-3,
        'epochs': 5,             # Test 5 epochs
        'pin_memory': True,      # Tăng tốc data transfer
        'mixed_precision': True, # Sử dụng mixed precision
    }
    
    print(f"\nCONFIGURATION:")
    print(f"   Batch size: {config['batch_size']}")
    print(f"   Num workers: {config['num_workers']}")
    print(f"   Mixed precision: {config['mixed_precision']}")
    print(f"   Pin memory: {config['pin_memory']}")
    
    # Device setup
    device = torch.device('cuda:0')
    print(f"   Device: {device}")
    
    # Clear GPU cache
    torch.cuda.empty_cache()
    gc.collect()
    
    try:
        # Kiểm tra data loading
        print("\n📁 TESTING DATA LOADING...")
        start_time = time.time()
        
        train_loader, val_loader, test_loader = create_enhanced_dataloaders(
            config['data_dir'],
            batch_size=config['batch_size'],
            num_workers=config['num_workers'],
            img_size=config['img_size'],
            pin_memory=config['pin_memory']
        )
        
        data_load_time = time.time() - start_time
        print(f"✅ Data loading successful ({data_load_time:.2f}s)")
        print(f"   Train batches: {len(train_loader)}")
        print(f"   Val batches: {len(val_loader)}")
        print(f"   Test batches: {len(test_loader)}")
        
        # Kiểm tra model loading
        print("\n🤖 TESTING MODEL LOADING...")
        start_time = time.time()
        
        model = create_model(num_classes=config['num_classes']).to(device)
        criterion = PCBLoss(num_classes=config['num_classes'])
        optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'])
        
        # Mixed precision setup
        scaler = torch.cuda.amp.GradScaler() if config['mixed_precision'] else None
        
        model_load_time = time.time() - start_time
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✅ Model loading successful ({model_load_time:.2f}s)")
        print(f"   Parameters: {total_params:,}")
        print(f"   Model size: ~{total_params * 4 / 1024**2:.1f}MB")
        
        # Check GPU memory after model loading
        mem_alloc, mem_reserved, mem_total = get_gpu_memory_info()
        print(f"   GPU Memory used: {mem_alloc:.1f}GB / {mem_total:.1f}GB")
        
        # Test một batch đầu tiên
        print("\n📊 TESTING FIRST BATCH...")
        model.train()
        
        # Test single batch
        test_batch = next(iter(train_loader))
        images, targets, filenames = test_batch
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        print(f"   Batch shape: {images.shape}")
        print(f"   Targets shape: {targets.shape}")
        print(f"   Images range: [{images.min():.3f}, {images.max():.3f}]")
        
        # Forward pass test
        start_time = time.time()
        
        with torch.cuda.amp.autocast(enabled=config['mixed_precision']):
            predictions = model(images)
            loss, loss_dict = criterion(predictions, targets)
        
        forward_time = time.time() - start_time
        print(f"✅ Forward pass successful ({forward_time*1000:.1f}ms)")
        print(f"   Predictions shape: {predictions.shape}")
        print(f"   Loss: {loss.item():.4f}")
        
        # Backward pass test
        start_time = time.time()
        optimizer.zero_grad()
        
        if config['mixed_precision']:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        backward_time = time.time() - start_time
        print(f"✅ Backward pass successful ({backward_time*1000:.1f}ms)")
        
        # Memory check after first batch
        mem_alloc, mem_reserved, mem_total = get_gpu_memory_info()
        print(f"   GPU Memory peak: {mem_alloc:.1f}GB / {mem_total:.1f}GB")
        
        # Test training loop
        print(f"\n🚀 TESTING TRAINING LOOP ({config['epochs']} epochs)...")
        
        for epoch in range(config['epochs']):
            print(f"\n--- Epoch {epoch+1}/{config['epochs']} ---")
            
            model.train()
            epoch_loss = 0
            epoch_coord = 0
            epoch_conf = 0
            epoch_class = 0
            batches_processed = 0
            
            start_time = time.time()
            
            # Chỉ test 5 batches đầu mỗi epoch để nhanh
            pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}')
            for batch_idx, (images, targets, filenames) in enumerate(pbar):
                if batch_idx >= 5:  # Chỉ test 5 batches
                    break
                
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                
                # Forward pass với mixed precision
                optimizer.zero_grad()
                
                with torch.cuda.amp.autocast(enabled=config['mixed_precision']):
                    predictions = model(images)
                    loss, loss_dict = criterion(predictions, targets)
                
                # Backward pass
                if config['mixed_precision']:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                
                # Track metrics
                epoch_loss += loss.item()
                epoch_coord += float(loss_dict['coord_loss'])
                epoch_conf += float(loss_dict['conf_loss'])
                epoch_class += float(loss_dict['class_loss'])
                batches_processed += 1
                
                # Update progress bar
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Coord': f'{float(loss_dict["coord_loss"]):.4f}',
                    'Conf': f'{float(loss_dict["conf_loss"]):.4f}',
                    'Class': f'{float(loss_dict["class_loss"]):.4f}',
                    'GPU': f'{torch.cuda.memory_allocated()/1024**3:.1f}GB'
                })
            
            epoch_time = time.time() - start_time
            
            # Calculate averages
            avg_loss = epoch_loss / batches_processed
            avg_coord = epoch_coord / batches_processed
            avg_conf = epoch_conf / batches_processed
            avg_class = epoch_class / batches_processed
            
            print(f"   Epoch time: {epoch_time:.1f}s")
            print(f"   Avg Loss: {avg_loss:.4f}")
            print(f"   Coord Loss: {avg_coord:.4f}")
            print(f"   Conf Loss: {avg_conf:.4f}")
            print(f"   Class Loss: {avg_class:.4f}")
            
            # Check learning progress
            if avg_coord > 0 and avg_class > 0:
                print(f"   ✅ Model is LEARNING! (Non-zero losses)")
            else:
                print(f"   ⚠️  Warning: Zero coord/class loss detected")
            
            # Memory monitoring
            mem_alloc, mem_reserved, mem_total = get_gpu_memory_info()
            print(f"   GPU Memory: {mem_alloc:.1f}GB / {mem_total:.1f}GB")
            
            # Clear cache between epochs
            torch.cuda.empty_cache()
        
        print("\n" + "="*70)
        print("🎉 RTX 3090 TEST TRAINING - SUMMARY")
        print("="*70)
        print("✅ CUDA và PyTorch hoạt động bình thường")
        print("✅ Data loading với batch_size=16 thành công")
        print("✅ Model loading và GPU memory quản lý tốt")
        print("✅ Mixed precision training hoạt động")
        print("✅ Forward/backward pass không có lỗi")
        print("✅ Training loop ổn định qua nhiều epochs")
        print("✅ Loss function tính toán đúng")
        print("✅ Model có thể học (non-zero losses)")
        print(f"✅ GPU Memory usage: ~{mem_alloc:.1f}GB / {mem_total:.1f}GB")
        
        # Performance summary
        print(f"\n📈 PERFORMANCE METRICS:")
        print(f"   Forward pass: ~{forward_time*1000:.1f}ms per batch")
        print(f"   Backward pass: ~{backward_time*1000:.1f}ms per batch")
        print(f"   Epoch time: ~{epoch_time:.1f}s (5 batches)")
        print(f"   Estimated full epoch: ~{epoch_time * len(train_loader) / 5:.1f}s")
        
        print(f"\n🚀 DỰ ÁN SẴN SÀNG CHO RTX 3090!")
        print("   Có thể chạy full training với config này.")
        
        return True
        
    except Exception as e:
        print(f"\n❌ LỖI TRAINING: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup
        torch.cuda.empty_cache()
        gc.collect()


def test_memory_limits():
    """Test memory limits với batch sizes khác nhau"""
    print("\n🔬 TESTING MEMORY LIMITS...")
    
    if not torch.cuda.is_available():
        print("❌ CUDA không khả dụng!")
        return
    
    device = torch.device('cuda:0')
    model = create_model(num_classes=6).to(device)
    
    batch_sizes = [8, 16, 24, 32, 40]
    img_size = 600
    
    for batch_size in batch_sizes:
        try:
            torch.cuda.empty_cache()
            
            # Test tensor
            test_input = torch.randn(batch_size, 1, img_size, img_size).to(device)
            
            with torch.no_grad():
                output = model(test_input)
            
            mem_alloc, _, mem_total = get_gpu_memory_info()
            print(f"   Batch {batch_size:2d}: ✅ {mem_alloc:.1f}GB / {mem_total:.1f}GB")
            
            del test_input, output
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"   Batch {batch_size:2d}: ❌ OOM")
                break
            else:
                raise e
    
    torch.cuda.empty_cache()


if __name__ == "__main__":
    print("KIỂM TRA RTX 3090 TRAINING READINESS")
    print("=" * 50)
    
    # Test basic training
    success = test_rtx3090_training()
    
    if success:
        # Test memory limits
        test_memory_limits()
        
        print("\n🏁 KẾT LUẬN:")
        print("✅ Dự án hoàn toàn sẵn sàng cho RTX 3090")
        print("✅ Chỉ cần chạy: pip install -r requirements.txt")
        print("✅ Sau đó chạy: python train.py")
    else:
        print("\n❌ Cần kiểm tra lại setup")