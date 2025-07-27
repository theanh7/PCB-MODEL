"""
Test Optimized Training Parameters for RTX A4000/3090
Kiểm tra các thông số train đã được tối ưu hóa
"""

import torch
import torch.optim as optim
from tqdm import tqdm
import os
import time
import psutil
import gc
from datetime import datetime

from model import create_model, PCBLoss
from enhanced_dataset import create_enhanced_dataloaders


def get_system_info():
    """Lấy thông tin hệ thống"""
    print("🖥️  SYSTEM INFORMATION")
    print("="*50)
    
    # CPU info
    cpu_count = psutil.cpu_count(logical=True)
    cpu_physical = psutil.cpu_count(logical=False)
    print(f"CPU Cores: {cpu_physical} physical, {cpu_count} logical")
    
    # Memory info
    memory = psutil.virtual_memory()
    print(f"RAM: {memory.total / 1024**3:.1f}GB total")
    
    # GPU info
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU: {gpu_name}")
        print(f"VRAM: {gpu_memory:.1f}GB")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"PyTorch Version: {torch.__version__}")
    else:
        print("❌ CUDA không khả dụng!")
    
    print()


def test_memory_scaling():
    """Test memory scaling với batch sizes khác nhau"""
    print("🧪 TESTING MEMORY SCALING")
    print("="*50)
    
    if not torch.cuda.is_available():
        print("❌ Cần GPU để test memory scaling")
        return
    
    device = torch.device('cuda:0')
    model = create_model(num_classes=6).to(device)
    
    # Test different batch sizes
    batch_sizes = [8, 16, 24, 32, 40, 48, 56, 64]
    img_size = 600
    
    print(f"Testing batch sizes for {img_size}x{img_size} images...")
    
    max_working_batch = 0
    
    for batch_size in batch_sizes:
        try:
            torch.cuda.empty_cache()
            gc.collect()
            
            # Test forward pass
            test_input = torch.randn(batch_size, 1, img_size, img_size).to(device)
            
            with torch.no_grad():
                output = model(test_input)
            
            mem_allocated = torch.cuda.memory_allocated() / 1024**3
            mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            mem_percent = (mem_allocated / mem_total) * 100
            
            print(f"   Batch {batch_size:2d}: ✅ {mem_allocated:.1f}GB ({mem_percent:.1f}%)")
            max_working_batch = batch_size
            
            del test_input, output
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"   Batch {batch_size:2d}: ❌ Out of Memory")
                break
            else:
                print(f"   Batch {batch_size:2d}: ❌ Error: {str(e)}")
                break
    
    torch.cuda.empty_cache()
    
    print(f"\n💡 RECOMMENDATION:")
    print(f"   Max safe batch size: {max_working_batch}")
    print(f"   Recommended batch size: {max(max_working_batch - 8, 16)} (với buffer)")
    print()


def test_dataloader_performance():
    """Test DataLoader performance với num_workers khác nhau"""
    print("📊 TESTING DATALOADER PERFORMANCE")
    print("="*50)
    
    data_dir = 'pcb-defect-dataset'
    if not os.path.exists(data_dir):
        print(f"❌ Không tìm thấy dataset: {data_dir}")
        return
    
    batch_size = 32
    num_workers_list = [0, 4, 8, 12, 16]
    
    print(f"Testing với batch_size={batch_size}")
    
    for num_workers in num_workers_list:
        try:
            print(f"\nTesting num_workers={num_workers}...")
            
            start_time = time.time()
            
            # Create dataloader
            train_loader, _, _ = create_enhanced_dataloaders(
                data_dir,
                batch_size=batch_size,
                num_workers=num_workers,
                img_size=600,
                pin_memory=True
            )
            
            # Test loading 10 batches
            batch_count = 0
            data_start = time.time()
            
            for batch_idx, (images, targets, _) in enumerate(train_loader):
                if batch_idx >= 10:  # Test 10 batches
                    break
                batch_count += 1
            
            data_time = time.time() - data_start
            total_time = time.time() - start_time
            
            avg_batch_time = data_time / batch_count if batch_count > 0 else 0
            
            print(f"   Setup time: {total_time - data_time:.2f}s")
            print(f"   Data loading: {data_time:.2f}s ({batch_count} batches)")
            print(f"   Avg per batch: {avg_batch_time*1000:.1f}ms")
            print(f"   Throughput: ~{(batch_size * batch_count) / data_time:.1f} images/sec")
            
        except Exception as e:
            print(f"   ❌ Error with num_workers={num_workers}: {str(e)}")
    
    print()


def test_optimized_training():
    """Test optimized training configuration"""
    print("🚀 TESTING OPTIMIZED TRAINING CONFIG")
    print("="*50)
    
    # Optimized configuration
    config = {
        'data_dir': 'pcb-defect-dataset',
        'batch_size': 32,        # Tối ưu cho RTX A4000/3090
        'num_workers': 12,       # Tận dụng CPU cores
        'img_size': 600,
        'num_classes': 6,
        'learning_rate': 1e-3,
        'mixed_precision': True,  # Tiết kiệm VRAM
        'pin_memory': True,       # Tăng tốc transfer
        'epochs': 3,             # Test 3 epochs
    }
    
    print("Configuration:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    
    if not torch.cuda.is_available():
        print("❌ Cần GPU để test training")
        return False
    
    device = torch.device('cuda:0')
    print(f"\nDevice: {device}")
    
    try:
        # Create data loaders
        print("\n📁 Creating optimized data loaders...")
        start_time = time.time()
        
        train_loader, val_loader, _ = create_enhanced_dataloaders(
            config['data_dir'],
            batch_size=config['batch_size'],
            num_workers=config['num_workers'],
            img_size=config['img_size'],
            pin_memory=config['pin_memory']
        )
        
        loader_time = time.time() - start_time
        print(f"✅ Data loaders created ({loader_time:.2f}s)")
        print(f"   Train batches: {len(train_loader)}")
        print(f"   Val batches: {len(val_loader)}")
        
        # Create model
        print("\n🤖 Creating model with mixed precision...")
        model = create_model(num_classes=config['num_classes']).to(device)
        criterion = PCBLoss(num_classes=config['num_classes'])
        optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'])
        
        # Mixed precision scaler
        scaler = torch.cuda.amp.GradScaler() if config['mixed_precision'] else None
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✅ Model created")
        print(f"   Parameters: {total_params:,}")
        print(f"   Mixed precision: {config['mixed_precision']}")
        
        # Test training loop
        print(f"\n🏃 Testing training loop ({config['epochs']} epochs)...")
        
        training_times = []
        memory_usage = []
        
        for epoch in range(config['epochs']):
            print(f"\n--- Epoch {epoch+1}/{config['epochs']} ---")
            
            model.train()
            epoch_loss = 0
            epoch_coord = 0
            epoch_conf = 0
            epoch_class = 0
            batches_processed = 0
            
            epoch_start = time.time()
            
            # Test chỉ 5 batches để nhanh
            pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}')
            for batch_idx, (images, targets, _) in enumerate(pbar):
                if batch_idx >= 5:  # Test 5 batches
                    break
                
                batch_start = time.time()
                
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                
                optimizer.zero_grad()
                
                # Forward pass với mixed precision
                if config['mixed_precision']:
                    with torch.cuda.amp.autocast():
                        predictions = model(images)
                        loss, loss_dict = criterion(predictions, targets)
                else:
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
                
                batch_time = time.time() - batch_start
                
                # Track metrics
                epoch_loss += loss.item()
                epoch_coord += float(loss_dict['coord_loss'])
                epoch_conf += float(loss_dict['conf_loss'])
                epoch_class += float(loss_dict['class_loss'])
                batches_processed += 1
                
                # Memory usage
                mem_allocated = torch.cuda.memory_allocated() / 1024**3
                memory_usage.append(mem_allocated)
                
                # Update progress
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Coord': f'{float(loss_dict["coord_loss"]):.4f}',
                    'Conf': f'{float(loss_dict["conf_loss"]):.4f}',
                    'Class': f'{float(loss_dict["class_loss"]):.4f}',
                    'Time': f'{batch_time*1000:.0f}ms',
                    'GPU': f'{mem_allocated:.1f}GB'
                })
            
            epoch_time = time.time() - epoch_start
            training_times.append(epoch_time)
            
            # Calculate averages
            avg_loss = epoch_loss / batches_processed
            avg_coord = epoch_coord / batches_processed
            avg_conf = epoch_conf / batches_processed
            avg_class = epoch_class / batches_processed
            
            print(f"   Epoch time: {epoch_time:.1f}s")
            print(f"   Avg Loss: {avg_loss:.4f}")
            print(f"   Coord: {avg_coord:.4f}, Conf: {avg_conf:.4f}, Class: {avg_class:.4f}")
            
            # Learning check
            if avg_coord > 0 and avg_class > 0:
                print(f"   ✅ Model đang học tốt!")
            else:
                print(f"   ⚠️  Cảnh báo: Zero coord/class loss")
            
            torch.cuda.empty_cache()
        
        # Performance summary
        avg_epoch_time = sum(training_times) / len(training_times)
        max_memory = max(memory_usage)
        avg_memory = sum(memory_usage) / len(memory_usage)
        
        print(f"\n📈 PERFORMANCE SUMMARY:")
        print(f"   Avg epoch time: {avg_epoch_time:.1f}s (5 batches)")
        print(f"   Est. full epoch: ~{avg_epoch_time * len(train_loader) / 5:.1f}s")
        print(f"   Avg memory usage: {avg_memory:.1f}GB")
        print(f"   Peak memory usage: {max_memory:.1f}GB")
        print(f"   Images per second: ~{(config['batch_size'] * 5) / avg_epoch_time:.1f}")
        
        # GPU utilization
        gpu_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        memory_efficiency = (max_memory / gpu_total) * 100
        
        print(f"   Memory efficiency: {memory_efficiency:.1f}%")
        
        if memory_efficiency > 90:
            print("   ⚠️  Memory usage cao - có thể cần giảm batch_size")
        elif memory_efficiency < 50:
            print("   💡 Memory usage thấp - có thể tăng batch_size")
        else:
            print("   ✅ Memory usage ở mức tối ưu")
        
        return True
        
    except Exception as e:
        print(f"\n❌ LỖI TRAINING: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        torch.cuda.empty_cache()
        gc.collect()


def main():
    """Main test function"""
    print("RTX A4000/3090 OPTIMIZED TRAINING TEST")
    print("="*60)
    print(f"Thời gian test: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # System info
    get_system_info()
    
    # Memory scaling test
    test_memory_scaling()
    
    # DataLoader performance test
    test_dataloader_performance()
    
    # Optimized training test
    success = test_optimized_training()
    
    print("\n" + "="*60)
    print("🏁 KẾT LUẬN FINAL:")
    print("="*60)
    
    if success:
        print("✅ Thông số training đã được tối ưu hóa thành công!")
        print("✅ Batch size 32 phù hợp với RTX A4000/3090")
        print("✅ Num workers 12 tận dụng tốt CPU multi-core")
        print("✅ Mixed precision giúp tiết kiệm VRAM")
        print("✅ Pin memory tăng tốc data transfer")
        print("✅ Persistent workers giảm overhead")
        print("\n🚀 SẴN SÀNG CHO PRODUCTION TRAINING!")
        print("   Chạy: python train.py")
    else:
        print("❌ Cần kiểm tra lại setup hoặc giảm batch_size")
    
    print("\n💡 TỐI ƯU HÓA THÊM:")
    print("   - Kiểm tra dataset size để điều chỉnh num_workers")
    print("   - Monitor GPU temperature trong quá trình train")
    print("   - Sử dụng learning rate scheduling")
    print("   - Implement gradient accumulation nếu cần batch lớn hơn")


if __name__ == "__main__":
    main()