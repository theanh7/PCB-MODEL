import os
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import cv2
from collections import defaultdict

from model import create_model
from enhanced_dataset import create_enhanced_dataloaders


class PCBEvaluator:
    """
    PCB Defect Detection Model Evaluator
    Provides comprehensive evaluation metrics and visualizations
    """
    
    def __init__(self, model_path, config, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config
        
        # Load model
        self.model = create_model(num_classes=config['num_classes']).to(self.device)
        self.load_model(model_path)
        
        # Class names
        self.class_names = {
            0: 'mouse_bite',
            1: 'spur', 
            2: 'missing_hole',
            3: 'short',
            4: 'open_circuit',
            5: 'spurious_copper'
        }
        
        # Evaluation metrics
        self.reset_metrics()
        
    def load_model(self, model_path):
        """Load trained model"""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        print(f"Model loaded from {model_path}")
        print(f"Best validation loss: {checkpoint.get('best_val_loss', 'N/A')}")
        
    def reset_metrics(self):
        """Reset evaluation metrics"""
        self.predictions = []
        self.ground_truths = []
        self.detection_results = []
        self.class_predictions = []
        self.class_ground_truths = []
        
    def calculate_iou(self, box1, box2):
        """Calculate IoU between two boxes"""
        # Convert from center format to corner format
        x1_min = box1[0] - box1[2] / 2
        y1_min = box1[1] - box1[3] / 2
        x1_max = box1[0] + box1[2] / 2
        y1_max = box1[1] + box1[3] / 2
        
        x2_min = box2[0] - box2[2] / 2
        y2_min = box2[1] - box2[3] / 2
        x2_max = box2[0] + box2[2] / 2
        y2_max = box2[1] + box2[3] / 2
        
        # Calculate intersection
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
            return 0.0
        
        intersection = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        
        # Calculate areas
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        
        # Calculate IoU
        union = area1 + area2 - intersection
        iou = intersection / (union + 1e-6)
        
        return iou
        
    def non_max_suppression(self, predictions, conf_threshold=0.5, iou_threshold=0.4):
        """Apply Non-Maximum Suppression to predictions"""
        if len(predictions) == 0:
            return []
        
        # Filter by confidence
        confident_preds = []
        for pred in predictions:
            if pred[4] >= conf_threshold:  # confidence score
                confident_preds.append(pred)
        
        if len(confident_preds) == 0:
            return []
        
        # Sort by confidence (descending)
        confident_preds.sort(key=lambda x: x[4], reverse=True)
        
        # Apply NMS
        keep = []
        while confident_preds:
            current = confident_preds.pop(0)
            keep.append(current)
            
            # Remove overlapping boxes
            confident_preds = [
                pred for pred in confident_preds
                if self.calculate_iou(current[:4], pred[:4]) < iou_threshold
            ]
        
        return keep
        
    def process_predictions(self, model_output, conf_threshold=0.5, iou_threshold=0.4):
        """Process raw model output to get final predictions"""
        B, A, features, H, W = model_output.shape
        
        all_predictions = []
        
        for b in range(B):
            batch_predictions = []
            
            for a in range(A):
                for h in range(H):
                    for w in range(W):
                        # Extract prediction components
                        x = model_output[b, a, 0, h, w].item()
                        y = model_output[b, a, 1, h, w].item()
                        width = model_output[b, a, 2, h, w].item()
                        height = model_output[b, a, 3, h, w].item()
                        confidence = model_output[b, a, 4, h, w].item()
                        
                        if confidence < conf_threshold:
                            continue
                        
                        # Get class probabilities
                        class_probs = model_output[b, a, 5:, h, w]
                        class_conf = torch.max(class_probs).item()
                        class_id = torch.argmax(class_probs).item()
                        
                        # Combined confidence
                        final_conf = confidence * class_conf
                        
                        if final_conf >= conf_threshold:
                            batch_predictions.append([x, y, width, height, final_conf, class_id])
            
            # Apply NMS
            final_predictions = self.non_max_suppression(
                batch_predictions, conf_threshold, iou_threshold
            )
            all_predictions.append(final_predictions)
        
        return all_predictions
        
    def evaluate_batch(self, images, targets, conf_threshold=0.5, iou_threshold=0.4):
        """Evaluate a single batch"""
        with torch.no_grad():
            # Get model predictions
            model_output = self.model(images)
            predictions = self.process_predictions(model_output, conf_threshold, iou_threshold)
            
            batch_size = len(predictions)
            
            for b in range(batch_size):
                pred_boxes = predictions[b]
                gt_boxes = targets[b][targets[b][:, 5] > 0]  # Valid ground truth boxes
                
                # Store for later analysis
                self.predictions.append(pred_boxes)
                self.ground_truths.append(gt_boxes.cpu().numpy())
                
                # Calculate detection metrics for this image
                image_results = self.calculate_detection_metrics(
                    pred_boxes, gt_boxes.cpu().numpy(), iou_threshold=0.5
                )
                self.detection_results.append(image_results)
                
                # Store class predictions for classification metrics
                for pred in pred_boxes:
                    self.class_predictions.append(int(pred[5]))
                
                for gt in gt_boxes:
                    self.class_ground_truths.append(int(gt[0]))
    
    def calculate_detection_metrics(self, predictions, ground_truths, iou_threshold=0.5):
        """Calculate detection metrics for a single image"""
        if len(ground_truths) == 0:
            return {
                'tp': len(predictions),  # All predictions are false positives
                'fp': len(predictions),
                'fn': 0,
                'precision': 0.0 if len(predictions) > 0 else 1.0,
                'recall': 1.0,  # No ground truth to miss
                'f1': 0.0 if len(predictions) > 0 else 1.0
            }
        
        if len(predictions) == 0:
            return {
                'tp': 0,
                'fp': 0,
                'fn': len(ground_truths),
                'precision': 1.0,  # No false positives
                'recall': 0.0,
                'f1': 0.0
            }
        
        # Match predictions to ground truths
        matched_gt = set()
        tp = 0
        fp = 0
        
        for pred in predictions:
            pred_box = pred[:4]
            pred_class = int(pred[5])
            
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, gt in enumerate(ground_truths):
                if gt_idx in matched_gt:
                    continue
                
                gt_box = gt[1:5]  # Skip class
                gt_class = int(gt[0])
                
                if pred_class == gt_class:
                    iou = self.calculate_iou(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
            
            if best_iou >= iou_threshold:
                tp += 1
                matched_gt.add(best_gt_idx)
            else:
                fp += 1
        
        fn = len(ground_truths) - len(matched_gt)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def evaluate_dataset(self, dataloader, conf_threshold=0.5, iou_threshold=0.4):
        """Evaluate the model on a dataset"""
        self.reset_metrics()
        
        self.model.eval()
        with torch.no_grad():
            for images, targets, _ in tqdm(dataloader, desc="Evaluating"):
                images = images.to(self.device)
                self.evaluate_batch(images, targets, conf_threshold, iou_threshold)
        
        return self.compute_final_metrics()
    
    def compute_final_metrics(self):
        """Compute final evaluation metrics"""
        # Overall detection metrics
        total_tp = sum(result['tp'] for result in self.detection_results)
        total_fp = sum(result['fp'] for result in self.detection_results)
        total_fn = sum(result['fn'] for result in self.detection_results)
        
        overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0
        
        # mAP calculation (simplified)
        precisions = [result['precision'] for result in self.detection_results]
        recalls = [result['recall'] for result in self.detection_results]
        f1_scores = [result['f1'] for result in self.detection_results]
        
        mAP = np.mean(precisions) if precisions else 0.0
        mean_recall = np.mean(recalls) if recalls else 0.0
        mean_f1 = np.mean(f1_scores) if f1_scores else 0.0
        
        # Class-wise metrics
        class_metrics = {}
        if self.class_predictions and self.class_ground_truths:
            # Ensure same length by truncating to minimum
            min_len = min(len(self.class_predictions), len(self.class_ground_truths))
            class_preds = self.class_predictions[:min_len]
            class_gts = self.class_ground_truths[:min_len]
            
            if min_len > 0:
                for class_id in range(self.config['num_classes']):
                    class_name = self.class_names[class_id]
                    
                    # Calculate per-class metrics
                    class_tp = sum(1 for p, g in zip(class_preds, class_gts) if p == g == class_id)
                    class_fp = sum(1 for p, g in zip(class_preds, class_gts) if p == class_id and g != class_id)
                    class_fn = sum(1 for p, g in zip(class_preds, class_gts) if p != class_id and g == class_id)
                    
                    class_precision = class_tp / (class_tp + class_fp) if (class_tp + class_fp) > 0 else 0.0
                    class_recall = class_tp / (class_tp + class_fn) if (class_tp + class_fn) > 0 else 0.0
                    class_f1 = 2 * class_precision * class_recall / (class_precision + class_recall) if (class_precision + class_recall) > 0 else 0.0
                    
                    class_metrics[class_name] = {
                        'precision': class_precision,
                        'recall': class_recall,
                        'f1': class_f1,
                        'support': class_gts.count(class_id)
                    }
        
        return {
            'overall': {
                'precision': overall_precision,
                'recall': overall_recall,
                'f1': overall_f1,
                'mAP': mAP,
                'mean_recall': mean_recall,
                'mean_f1': mean_f1,
                'total_tp': total_tp,
                'total_fp': total_fp,
                'total_fn': total_fn
            },
            'class_wise': class_metrics
        }
    
    def plot_confusion_matrix(self, save_path=None):
        """Plot confusion matrix"""
        if not self.class_predictions or not self.class_ground_truths:
            print("No classification data available for confusion matrix")
            return
        
        # Ensure same length
        min_len = min(len(self.class_predictions), len(self.class_ground_truths))
        y_pred = self.class_predictions[:min_len]
        y_true = self.class_ground_truths[:min_len]
        
        if min_len == 0:
            print("No data available for confusion matrix")
            return
        
        cm = confusion_matrix(y_true, y_pred, labels=list(range(self.config['num_classes'])))
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=[self.class_names[i] for i in range(self.config['num_classes'])],
            yticklabels=[self.class_names[i] for i in range(self.config['num_classes'])]
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def print_evaluation_report(self, metrics):
        """Print detailed evaluation report"""
        print("\n" + "="*80)
        print("PCB DEFECT DETECTION EVALUATION REPORT")
        print("="*80)
        
        # Overall metrics
        overall = metrics['overall']
        print(f"\nOVERALL DETECTION METRICS:")
        print(f"  Precision: {overall['precision']:.4f}")
        print(f"  Recall:    {overall['recall']:.4f}")
        print(f"  F1-Score:  {overall['f1']:.4f}")
        print(f"  mAP:       {overall['mAP']:.4f}")
        print(f"  True Positives:  {overall['total_tp']}")
        print(f"  False Positives: {overall['total_fp']}")
        print(f"  False Negatives: {overall['total_fn']}")
        
        # Class-wise metrics
        if metrics['class_wise']:
            print(f"\nCLASS-WISE METRICS:")
            print(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
            print("-" * 60)
            
            for class_name, class_metrics in metrics['class_wise'].items():
                print(f"{class_name:<15} "
                      f"{class_metrics['precision']:<10.4f} "
                      f"{class_metrics['recall']:<10.4f} "
                      f"{class_metrics['f1']:<10.4f} "
                      f"{class_metrics['support']:<10}")
        
        print("="*80)


def main():
    """Main evaluation function"""
    # Configuration
    config = {
        'data_dir': 'pcb-defect-dataset',
        'model_path': 'outputs/best_model.pth',
        'batch_size': 32,
        'num_workers': 12,
        'img_size': 600,
        'num_classes': 6,
        'conf_threshold': 0.5,
        'iou_threshold': 0.4,
        'output_dir': 'evaluation_results'
    }
    
    # Create output directory
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Check if model exists
    if not os.path.exists(config['model_path']):
        print(f"Model not found: {config['model_path']}")
        print("Please train the model first using train.py")
        return
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader, test_loader = create_enhanced_dataloaders(
        config['data_dir'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        img_size=config['img_size'],
        pin_memory=True
    )
    
    # Create evaluator
    evaluator = PCBEvaluator(config['model_path'], config)
    
    # Evaluate on validation set
    print("Evaluating on validation set...")
    val_metrics = evaluator.evaluate_dataset(
        val_loader, 
        conf_threshold=config['conf_threshold'],
        iou_threshold=config['iou_threshold']
    )
    
    # Print validation results
    print("\nVALIDATION SET RESULTS:")
    evaluator.print_evaluation_report(val_metrics)
    
    # Plot confusion matrix for validation
    evaluator.plot_confusion_matrix(
        save_path=os.path.join(config['output_dir'], 'confusion_matrix_val.png')
    )
    
    # Evaluate on test set
    print("Evaluating on test set...")
    test_metrics = evaluator.evaluate_dataset(
        test_loader,
        conf_threshold=config['conf_threshold'],
        iou_threshold=config['iou_threshold']
    )
    
    # Print test results
    print("\nTEST SET RESULTS:")
    evaluator.print_evaluation_report(test_metrics)
    
    # Plot confusion matrix for test
    evaluator.plot_confusion_matrix(
        save_path=os.path.join(config['output_dir'], 'confusion_matrix_test.png')
    )
    
    print(f"\nEvaluation completed! Results saved to {config['output_dir']}")


if __name__ == "__main__":
    main()