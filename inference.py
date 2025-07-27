import os
import torch
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import argparse
from pathlib import Path

from model import create_model


class PCBInference:
    """
    PCB Defect Detection Inference Pipeline
    Optimized for direct numpy array input: (1, 1, H, W), dtype=float32
    """
    
    def __init__(self, model_path, device=None, conf_threshold=0.5, iou_threshold=0.4):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # Class names and colors
        self.class_names = {
            0: 'mouse_bite',
            1: 'spur', 
            2: 'missing_hole',
            3: 'short',
            4: 'open_circuit',
            5: 'spurious_copper'
        }
        
        # Colors for each class (BGR format for OpenCV)
        self.class_colors = {
            0: (0, 255, 255),      # Yellow - mouse_bite
            1: (255, 0, 255),      # Magenta - spur
            2: (0, 0, 255),        # Red - missing_hole
            3: (255, 255, 0),      # Cyan - short
            4: (0, 255, 0),        # Green - open_circuit
            5: (255, 0, 0),        # Blue - spurious_copper
        }
        
        # Load model
        self.model = self.load_model(model_path)
        
        print(f"Inference pipeline initialized on {self.device}")
        print(f"Expected input: np.ndarray, dtype=float32, shape=(1, 1, H, W)")
        print(f"Confidence threshold: {conf_threshold}")
        print(f"IoU threshold: {iou_threshold}")
    
    def load_model(self, model_path):
        """Load trained model"""
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Get model config from checkpoint
        config = checkpoint.get('config', {'num_classes': 6})
        
        # Create model
        model = create_model(num_classes=config['num_classes']).to(self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print(f"Model loaded from {model_path}")
        print(f"Best validation loss: {checkpoint.get('best_val_loss', 'N/A')}")
        
        return model
    
    def validate_input(self, input_array):
        """Validate input array format"""
        if not isinstance(input_array, np.ndarray):
            raise TypeError(f"Expected np.ndarray, got {type(input_array)}")
        
        if input_array.dtype != np.float32:
            raise TypeError(f"Expected dtype=float32, got {input_array.dtype}")
        
        if len(input_array.shape) != 4:
            raise ValueError(f"Expected 4D array (1, 1, H, W), got shape {input_array.shape}")
        
        if input_array.shape[0] != 1 or input_array.shape[1] != 1:
            raise ValueError(f"Expected shape (1, 1, H, W), got {input_array.shape}")
        
        # Check value range (should be normalized)
        if input_array.min() < -2.0 or input_array.max() > 2.0:
            print(f"Warning: Input values outside expected range [-1, 1]. Min: {input_array.min():.3f}, Max: {input_array.max():.3f}")
        
        return True
    
    def preprocess_from_path(self, image_path, target_size=600):
        """
        Preprocess image from file path to required format
        Returns: np.ndarray, dtype=float32, shape=(1, 1, target_size, target_size)
        """
        # Load image as grayscale
        if isinstance(image_path, str):
            image = Image.open(image_path).convert('L')
        else:
            image = image_path.convert('L')
        
        original_size = image.size
        
        # Resize to target size
        image = image.resize((target_size, target_size), Image.BILINEAR)
        
        # Convert to numpy array
        image_np = np.array(image, dtype=np.float32)
        
        # Normalize to [-1, 1] range (matching training normalization)
        image_np = (image_np / 255.0 - 0.5) / 0.5
        
        # Add batch and channel dimensions: (H, W) -> (1, 1, H, W)
        input_array = image_np[np.newaxis, np.newaxis, :, :]
        
        return input_array, original_size
    
    def preprocess_from_array(self, input_array, original_size=None):
        """
        Validate and convert input array to torch tensor
        Input: np.ndarray, dtype=float32, shape=(1, 1, H, W)
        """
        # Validate input format
        self.validate_input(input_array)
        
        # Convert to torch tensor
        input_tensor = torch.from_numpy(input_array).to(self.device)
        
        # Get original size if not provided
        if original_size is None:
            H, W = input_array.shape[2], input_array.shape[3]
            original_size = (W, H)  # PIL format (width, height)
        
        return input_tensor, original_size
    
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
    
    def non_max_suppression(self, predictions):
        """Apply Non-Maximum Suppression"""
        if len(predictions) == 0:
            return []
        
        # Filter by confidence
        confident_preds = [pred for pred in predictions if pred[4] >= self.conf_threshold]
        
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
                if self.calculate_iou(current[:4], pred[:4]) < self.iou_threshold
            ]
        
        return keep
    
    def postprocess_predictions(self, model_output):
        """Process raw model output to get final predictions"""
        B, A, features, H, W = model_output.shape
        
        predictions = []
        
        for a in range(A):
            for h in range(H):
                for w in range(W):
                    # Extract prediction components (assuming inference mode)
                    x = model_output[0, a, 0, h, w].item()
                    y = model_output[0, a, 1, h, w].item()
                    width = model_output[0, a, 2, h, w].item()
                    height = model_output[0, a, 3, h, w].item()
                    confidence = model_output[0, a, 4, h, w].item()
                    
                    if confidence < self.conf_threshold:
                        continue
                    
                    # Get class probabilities
                    class_probs = model_output[0, a, 5:, h, w]
                    class_conf = torch.max(class_probs).item()
                    class_id = torch.argmax(class_probs).item()
                    
                    # Combined confidence
                    final_conf = confidence * class_conf
                    
                    if final_conf >= self.conf_threshold:
                        predictions.append([x, y, width, height, final_conf, class_id])
        
        # Apply NMS
        final_predictions = self.non_max_suppression(predictions)
        
        return final_predictions
    
    def scale_predictions(self, predictions, original_size, model_input_size=600):
        """Scale predictions back to original image size"""
        if len(predictions) == 0:
            return []
        
        width_scale = original_size[0] / model_input_size
        height_scale = original_size[1] / model_input_size
        
        scaled_predictions = []
        for pred in predictions:
            x, y, w, h, conf, cls = pred
            
            # Scale coordinates
            scaled_x = x * width_scale
            scaled_y = y * height_scale
            scaled_w = w * width_scale
            scaled_h = h * height_scale
            
            scaled_predictions.append([scaled_x, scaled_y, scaled_w, scaled_h, conf, cls])
        
        return scaled_predictions
    
    def predict_from_array(self, input_array, original_size=None, return_raw=False):
        """
        Main prediction function for preprocessed array input
        
        Args:
            input_array: np.ndarray, dtype=float32, shape=(1, 1, H, W)
            original_size: tuple (width, height) for scaling predictions back
            return_raw: bool, if True return raw model output
            
        Returns:
            predictions: list of [x, y, w, h, confidence, class_id]
            raw_output: torch tensor (if return_raw=True)
        """
        # Validate and convert input
        input_tensor, original_size = self.preprocess_from_array(input_array, original_size)
        
        # Inference
        with torch.no_grad():
            model_output = self.model(input_tensor)
        
        if return_raw:
            return model_output
        
        # Postprocess
        predictions = self.postprocess_predictions(model_output)
        
        # Scale predictions to original size
        if original_size:
            model_input_size = input_array.shape[2]  # Assuming square input
            scaled_predictions = self.scale_predictions(predictions, original_size, model_input_size)
        else:
            scaled_predictions = predictions
        
        return scaled_predictions
    
    def predict_from_path(self, image_path, save_result=True, output_dir="inference_results"):
        """
        Predict defects from image file path
        This method handles the full pipeline: load -> preprocess -> predict -> visualize
        """
        # Preprocess from path
        input_array, original_size = self.preprocess_from_path(image_path)
        
        # Predict
        predictions = self.predict_from_array(input_array, original_size)
        
        # Load original image for visualization
        original_image = Image.open(image_path).convert('RGB')
        
        # Draw results
        result_image = self.draw_predictions(original_image, predictions)
        
        # Save result
        if save_result:
            os.makedirs(output_dir, exist_ok=True)
            image_name = Path(image_path).stem
            save_path = os.path.join(output_dir, f"{image_name}_result.jpg")
            result_image.save(save_path)
        
        # Print results
        print(f"\nDetection Results for {Path(image_path).name}:")
        print(f"Found {len(predictions)} defects:")
        
        for i, pred in enumerate(predictions):
            x, y, w, h, conf, cls = pred
            class_name = self.class_names[int(cls)]
            print(f"  {i+1}. {class_name} (confidence: {conf:.3f}) at ({x:.1f}, {y:.1f})")
        
        return predictions, result_image
    
    def draw_predictions(self, image, predictions, save_path=None):
        """Draw predictions on image"""
        # Convert to PIL for drawing
        if isinstance(image, np.ndarray):
            if len(image.shape) == 2:  # Grayscale
                image_pil = Image.fromarray(image).convert('RGB')
            else:
                image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            image_pil = image.convert('RGB')
        
        draw = ImageDraw.Draw(image_pil)
        
        # Try to load a font
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        for pred in predictions:
            x, y, w, h, conf, cls = pred
            cls = int(cls)
            
            # Convert from center format to corner format
            x1 = x - w / 2
            y1 = y - h / 2
            x2 = x + w / 2
            y2 = y + h / 2
            
            # Get class info
            class_name = self.class_names[cls]
            color = self.class_colors[cls]
            
            # Draw bounding box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            
            # Draw label
            label = f"{class_name}: {conf:.2f}"
            
            # Get text size
            bbox = draw.textbbox((0, 0), label, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # Draw label background
            draw.rectangle(
                [x1, y1 - text_height - 4, x1 + text_width + 4, y1],
                fill=color
            )
            
            # Draw label text
            draw.text((x1 + 2, y1 - text_height - 2), label, fill=(255, 255, 255), font=font)
        
        if save_path:
            image_pil.save(save_path)
            print(f"Result saved to {save_path}")
        
        return image_pil


def create_sample_input(height=600, width=600, normalize=True):
    """
    Create sample input array for testing
    Returns: np.ndarray, dtype=float32, shape=(1, 1, H, W)
    """
    # Create random grayscale image
    sample_image = np.random.randint(0, 256, (height, width), dtype=np.uint8)
    
    if normalize:
        # Normalize to [-1, 1] range (matching training)
        sample_array = (sample_image.astype(np.float32) / 255.0 - 0.5) / 0.5
    else:
        # Keep as [0, 1] range
        sample_array = sample_image.astype(np.float32) / 255.0
    
    # Add batch and channel dimensions
    input_array = sample_array[np.newaxis, np.newaxis, :, :]
    
    return input_array


def main():
    """Main inference function with array input example"""
    parser = argparse.ArgumentParser(description='PCB Defect Detection Inference (Optimized)')
    parser.add_argument('--model', type=str, default='outputs/best_model.pth',
                      help='Path to trained model')
    parser.add_argument('--input', type=str, required=False,
                      help='Path to input image (optional)')
    parser.add_argument('--test_array', action='store_true',
                      help='Test with sample array input')
    parser.add_argument('--conf_threshold', type=float, default=0.5,
                      help='Confidence threshold for detections')
    parser.add_argument('--iou_threshold', type=float, default=0.4,
                      help='IoU threshold for NMS')
    parser.add_argument('--device', type=str, default=None,
                      help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Model not found: {args.model}")
        print("Please train the model first using train.py")
        return
    
    # Create inference pipeline
    device = torch.device(args.device) if args.device else None
    inference = PCBInference(
        model_path=args.model,
        device=device,
        conf_threshold=args.conf_threshold,
        iou_threshold=args.iou_threshold
    )
    
    if args.test_array:
        # Test with sample array input
        print("\nTesting with sample array input...")
        sample_input = create_sample_input(600, 600, normalize=True)
        
        print(f"Sample input shape: {sample_input.shape}")
        print(f"Sample input dtype: {sample_input.dtype}")
        print(f"Sample input range: [{sample_input.min():.3f}, {sample_input.max():.3f}]")
        
        # Run inference
        predictions = inference.predict_from_array(sample_input, original_size=(600, 600))
        
        print(f"Predictions: {len(predictions)} detections")
        for i, pred in enumerate(predictions):
            x, y, w, h, conf, cls = pred
            class_name = inference.class_names[int(cls)]
            print(f"  {i+1}. {class_name} (confidence: {conf:.3f}) at ({x:.1f}, {y:.1f})")
    
    elif args.input:
        # Test with image file
        if os.path.isfile(args.input):
            inference.predict_from_path(args.input)
        else:
            print(f"Input file not found: {args.input}")
    else:
        print("Please provide --input or use --test_array for testing")
        print("\nExample usage:")
        print("python inference_optimized.py --model best_model.pth --test_array")
        print("python inference_optimized.py --model best_model.pth --input test.jpg")


if __name__ == "__main__":
    main()