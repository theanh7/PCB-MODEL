import os
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import logging
from datetime import datetime

from model import create_model, PCBLoss
from enhanced_dataset import create_enhanced_dataloaders


class PCBTrainer:
    """
    PCB Defect Detection Model Trainer
    Optimized for Basler aca3800 10gm camera grayscale images
    """
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup logging
        self.setup_logging()
        
        # Create model
        self.model = create_model(
            num_classes=config['num_classes'],
            pretrained=config['pretrained']
        ).to(self.device)
        
        # Create loss function
        self.criterion = PCBLoss(
            num_classes=config['num_classes'],
            lambda_coord=config['lambda_coord'],
            lambda_noobj=config['lambda_noobj']
        )
        
        # Create optimizer - Adam with weight decay for better generalization
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay'],
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler
        if config['scheduler'] == 'reduce_on_plateau':
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=10,
                min_lr=1e-6
            )
        elif config['scheduler'] == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=config['epochs'],
                eta_min=1e-6
            )
        else:
            self.scheduler = None
        
        # Mixed precision training setup
        self.mixed_precision = config.get('mixed_precision', False)
        self.scaler = torch.cuda.amp.GradScaler() if self.mixed_precision else None
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        self.best_val_loss = float('inf')
        
        # Create output directory
        os.makedirs(config['output_dir'], exist_ok=True)
        
    def setup_logging(self):
        """Setup logging configuration"""
        log_file = os.path.join(
            self.config['output_dir'],
            f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def train_epoch(self, dataloader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        total_coord_loss = 0.0
        total_conf_loss = 0.0
        total_class_loss = 0.0
        
        pbar = tqdm(dataloader, desc='Training')
        for batch_idx, (images, targets, _) in enumerate(pbar):
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            
            # Forward pass with mixed precision
            self.optimizer.zero_grad()
            
            if self.mixed_precision:
                with torch.cuda.amp.autocast():
                    predictions = self.model(images)
                    loss, loss_dict = self.criterion(predictions, targets)
            else:
                predictions = self.model(images)
                loss, loss_dict = self.criterion(predictions, targets)
            
            # Backward pass with mixed precision
            if self.mixed_precision:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient clipping for stability
            if self.mixed_precision:
                self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Optimizer step with mixed precision
            if self.mixed_precision:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            total_coord_loss += loss_dict['coord_loss']
            total_conf_loss += loss_dict['conf_loss']
            total_class_loss += loss_dict['class_loss']
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'LR': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
        
        avg_loss = total_loss / len(dataloader)
        avg_coord_loss = total_coord_loss / len(dataloader)
        avg_conf_loss = total_conf_loss / len(dataloader)
        avg_class_loss = total_class_loss / len(dataloader)
        
        return {
            'total_loss': avg_loss,
            'coord_loss': avg_coord_loss,
            'conf_loss': avg_conf_loss,
            'class_loss': avg_class_loss
        }
    
    def validate_epoch(self, dataloader):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        total_coord_loss = 0.0
        total_conf_loss = 0.0
        total_class_loss = 0.0
        
        with torch.no_grad():
            pbar = tqdm(dataloader, desc='Validation')
            for images, targets, _ in pbar:
                images = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                
                # Forward pass with mixed precision
                if self.mixed_precision:
                    with torch.cuda.amp.autocast():
                        predictions = self.model(images)
                        loss, loss_dict = self.criterion(predictions, targets)
                else:
                    predictions = self.model(images)
                    loss, loss_dict = self.criterion(predictions, targets)
                
                
                # Update metrics
                total_loss += loss.item()
                total_coord_loss += loss_dict['coord_loss']
                total_conf_loss += loss_dict['conf_loss']
                total_class_loss += loss_dict['class_loss']
                
                pbar.set_postfix({'Val Loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(dataloader)
        avg_coord_loss = total_coord_loss / len(dataloader)
        avg_conf_loss = total_conf_loss / len(dataloader)
        avg_class_loss = total_class_loss / len(dataloader)
        
        return {
            'total_loss': avg_loss,
            'coord_loss': avg_coord_loss,
            'conf_loss': avg_conf_loss,
            'class_loss': avg_class_loss
        }
    
    def train(self, train_loader, val_loader):
        """Main training loop"""
        self.logger.info("Starting training...")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(self.config['epochs']):
            self.logger.info(f"\nEpoch {epoch+1}/{self.config['epochs']}")
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            self.train_losses.append(train_metrics['total_loss'])
            
            # Validate
            val_metrics = self.validate_epoch(val_loader)
            self.val_losses.append(val_metrics['total_loss'])
            
            # Learning rate scheduling
            current_lr = self.optimizer.param_groups[0]['lr']
            self.learning_rates.append(current_lr)
            
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['total_loss'])
                else:
                    self.scheduler.step()
            
            # Log metrics
            self.logger.info(f"Train Loss: {train_metrics['total_loss']:.4f}")
            self.logger.info(f"  - Coord: {train_metrics['coord_loss']:.4f}")
            self.logger.info(f"  - Conf: {train_metrics['conf_loss']:.4f}")
            self.logger.info(f"  - Class: {train_metrics['class_loss']:.4f}")
            
            self.logger.info(f"Val Loss: {val_metrics['total_loss']:.4f}")
            self.logger.info(f"  - Coord: {val_metrics['coord_loss']:.4f}")
            self.logger.info(f"  - Conf: {val_metrics['conf_loss']:.4f}")
            self.logger.info(f"  - Class: {val_metrics['class_loss']:.4f}")
            
            self.logger.info(f"Learning Rate: {current_lr:.6f}")
            
            # Save best model
            if val_metrics['total_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['total_loss']
                self.save_checkpoint(epoch, is_best=True)
                self.logger.info(f"New best model saved! Val Loss: {self.best_val_loss:.4f}")
            
            # Save regular checkpoint
            if (epoch + 1) % self.config['save_every'] == 0:
                self.save_checkpoint(epoch, is_best=False)
            
            # Plot training curves
            if (epoch + 1) % 10 == 0:
                self.plot_training_curves()
        
        self.logger.info("Training completed!")
        self.plot_training_curves()
        
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates,
            'config': self.config
        }
        
        if is_best:
            filename = os.path.join(self.config['output_dir'], 'best_model.pth')
        else:
            filename = os.path.join(self.config['output_dir'], f'checkpoint_epoch_{epoch+1}.pth')
        
        torch.save(checkpoint, filename)
        
    def plot_training_curves(self):
        """Plot training and validation curves"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(self.train_losses) + 1)
        
        # Loss curves
        ax1.plot(epochs, self.train_losses, 'b-', label='Train Loss')
        ax1.plot(epochs, self.val_losses, 'r-', label='Val Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Learning rate
        ax2.plot(epochs, self.learning_rates, 'g-')
        ax2.set_title('Learning Rate Schedule')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Learning Rate')
        ax2.set_yscale('log')
        ax2.grid(True)
        
        # Loss ratio
        if len(self.val_losses) > 0 and len(self.train_losses) > 0:
            loss_ratio = [v/t if t > 0 else 1.0 for v, t in zip(self.val_losses, self.train_losses)]
            ax3.plot(epochs, loss_ratio, 'purple')
            ax3.set_title('Validation/Training Loss Ratio')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Val Loss / Train Loss')
            ax3.axhline(y=1.0, color='r', linestyle='--', alpha=0.5)
            ax3.grid(True)
        
        # Moving average of losses
        if len(self.train_losses) >= 5:
            window = 5
            train_ma = np.convolve(self.train_losses, np.ones(window)/window, mode='valid')
            val_ma = np.convolve(self.val_losses, np.ones(window)/window, mode='valid')
            epochs_ma = range(window, len(self.train_losses) + 1)
            
            ax4.plot(epochs_ma, train_ma, 'b-', label=f'Train MA({window})')
            ax4.plot(epochs_ma, val_ma, 'r-', label=f'Val MA({window})')
            ax4.set_title('Moving Average of Losses')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Loss')
            ax4.legend()
            ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config['output_dir'], 'training_curves.png'), dpi=300)
        plt.close()


def main():
    """Main training function"""
    # Training configuration - Optimized for RTX A4000/3090
    config = {
        # Data parameters
        'data_dir': 'pcb-defect-dataset',
        'batch_size': 32,        # Optimized for RTX A4000/3090 (16-24GB VRAM)
        'num_workers': 12,       # Tận dụng CPU multi-core (8-16 cores)
        'img_size': 600,
        
        # Model parameters
        'num_classes': 6,
        'pretrained': True,
        
        # Loss parameters
        'lambda_coord': 5.0,
        'lambda_noobj': 0.5,
        
        # Training parameters - Optimized for high-end GPUs
        'epochs': 100,
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'scheduler': 'reduce_on_plateau',  # 'reduce_on_plateau', 'cosine', or None
        'mixed_precision': True,           # Enable mixed precision for RTX A4000/3090
        'pin_memory': True,                # Faster data transfer to GPU
        
        # Saving parameters
        'output_dir': 'outputs',
        'save_every': 10,
    }
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader, test_loader = create_enhanced_dataloaders(
        config['data_dir'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        img_size=config['img_size'],
        pin_memory=config['pin_memory']
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Create trainer
    trainer = PCBTrainer(config)
    
    # Start training
    trainer.train(train_loader, val_loader)
    
    print("Training completed! Check the outputs directory for results.")


if __name__ == "__main__":
    main()