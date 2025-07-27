import torch
import torch.nn as nn
import torch.nn.functional as F


class PCBDefectNet(nn.Module):
    """
    Optimized PCB defect detection model for grayscale images.
    Designed specifically for Basler aca3800 10gm camera output.
    """
    
    def __init__(self, num_classes=6, input_size=600, anchors_per_cell=3):
        super(PCBDefectNet, self).__init__()
        self.num_classes = num_classes
        self.input_size = input_size
        self.anchors_per_cell = anchors_per_cell
        
        # Backbone: Efficient feature extraction for PCB patterns
        self.backbone = nn.Sequential(
            # Initial conv for grayscale processing
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # Block 1: 600x600 -> 300x300
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Block 2: 300x300 -> 150x150
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Block 3: 150x150 -> 75x75
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Block 4: 75x75 -> 37x37 (good for small defects)
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            # Block 5: 37x37 -> 19x19 (final detection grid)
            nn.Conv2d(512, 512, 3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        
        # Detection head: outputs bounding boxes + classes
        # Each cell predicts: (x, y, w, h, confidence, class_probs) * anchors_per_cell
        output_size = 5 + num_classes  # x,y,w,h,conf + 6 classes
        self.detection_head = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, anchors_per_cell * output_size, 1)
        )
        
        # Define anchor boxes optimized for PCB defects (relative to 19x19 grid)
        self.register_buffer('anchors', torch.tensor([
            [0.5, 0.5],   # Small defects (missing holes, shorts)
            [1.0, 1.0],   # Medium defects (mouse bites, spurs)
            [2.0, 2.0],   # Large defects (spurious copper, open circuits)
        ]))
        
        self.grid_size = 19  # Final grid size after backbone
        
    def forward(self, x):
        # Extract features
        features = self.backbone(x)  # [B, 512, 19, 19]
        
        # Get detection predictions
        detections = self.detection_head(features)  # [B, anchors*(5+classes), 19, 19]
        
        B, _, H, W = detections.shape
        
        # Reshape to [B, anchors, 5+classes, H, W]
        detections = detections.view(B, self.anchors_per_cell, 5 + self.num_classes, H, W)
        
        # Apply activations
        xy = torch.sigmoid(detections[:, :, :2])  # x,y coordinates [0,1]
        wh = detections[:, :, 2:4]  # width, height (will be transformed)
        conf = torch.sigmoid(detections[:, :, 4:5])  # confidence [0,1]
        classes = detections[:, :, 5:]  # class logits
        
        if self.training:
            return torch.cat([xy, wh, conf, classes], dim=2)
        else:
            # Convert to absolute coordinates for inference
            grid_x, grid_y = torch.meshgrid(
                torch.arange(W, device=x.device),
                torch.arange(H, device=x.device),
                indexing='xy'
            )
            
            # Convert relative coordinates to absolute
            pred_x = (xy[:, :, 0] + grid_x) / W
            pred_y = (xy[:, :, 1] + grid_y) / H
            
            # Convert width/height using anchors
            anchor_w = self.anchors[:, 0].view(1, -1, 1, 1)
            anchor_h = self.anchors[:, 1].view(1, -1, 1, 1)
            pred_w = torch.exp(wh[:, :, 0]) * anchor_w / W
            pred_h = torch.exp(wh[:, :, 1]) * anchor_h / H
            
            pred_boxes = torch.stack([pred_x, pred_y, pred_w, pred_h], dim=2)
            pred_classes = torch.softmax(classes, dim=2)
            
            return torch.cat([pred_boxes, conf, pred_classes], dim=2)


class PCBLoss(nn.Module):
    """
    Custom loss function for PCB defect detection.
    Combines coordinate regression, confidence, and classification losses.
    """
    
    def __init__(self, num_classes=6, lambda_coord=5.0, lambda_noobj=0.5):
        super(PCBLoss, self).__init__()
        self.num_classes = num_classes
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        
    def forward(self, predictions, targets):
        """
        Args:
            predictions: [B, anchors, 5+classes, H, W]
            targets: [B, max_objects, 6] (class, x, y, w, h, object_mask)
        """
        B, A, _, H, W = predictions.shape
        device = predictions.device
        
        # Extract prediction components
        pred_xy = predictions[:, :, :2]  # [B, A, 2, H, W]
        pred_wh = predictions[:, :, 2:4]  # [B, A, 2, H, W]
        pred_conf = predictions[:, :, 4]  # [B, A, H, W]
        pred_classes = predictions[:, :, 5:]  # [B, A, classes, H, W]
        
        # Initialize loss components
        coord_loss = 0.0
        conf_loss = 0.0
        class_loss = 0.0
        
        for b in range(B):
            batch_targets = targets[b]
            valid_targets = batch_targets[batch_targets[:, 5] > 0]  # Filter valid objects
            
            if len(valid_targets) == 0:
                # No objects, only no-object confidence loss
                conf_loss += F.binary_cross_entropy(
                    pred_conf[b].flatten(),
                    torch.zeros_like(pred_conf[b].flatten())
                )
                continue
            
            # Create responsibility masks
            obj_mask = torch.zeros(A, H, W, device=device)
            target_boxes = torch.zeros(A, H, W, 4, device=device)
            target_classes = torch.zeros(A, H, W, self.num_classes, device=device)
            
            for target in valid_targets:
                cls, x, y, w, h = target[:5]
                
                # Convert to grid coordinates
                grid_x = int(x * W)
                grid_y = int(y * H)
                grid_x = min(grid_x, W - 1)
                grid_y = min(grid_y, H - 1)
                
                # Find best anchor based on IoU with target
                target_wh = torch.tensor([w, h], device=device)
                anchor_ious = []
                
                for a in range(A):
                    anchor_wh = torch.tensor([0.1, 0.1], device=device)  # Default anchor
                    iou = self.calculate_iou(anchor_wh, target_wh)
                    anchor_ious.append(iou)
                
                best_anchor = torch.argmax(torch.tensor(anchor_ious))
                
                # Assign responsibility
                obj_mask[best_anchor, grid_y, grid_x] = 1
                target_boxes[best_anchor, grid_y, grid_x] = torch.tensor([x, y, w, h])
                target_classes[best_anchor, grid_y, grid_x, int(cls)] = 1
            
            # Calculate losses
            responsible_mask = obj_mask > 0
            
            if responsible_mask.sum() > 0:
                # Coordinate loss (only for responsible predictions)
                # pred_xy[b] shape: [A, 2, H, W], need [A, H, W, 2] then mask
                xy_pred = pred_xy[b].permute(0, 2, 3, 1)[responsible_mask]  # [num_responsible, 2]
                wh_pred = pred_wh[b].permute(0, 2, 3, 1)[responsible_mask]  # [num_responsible, 2]
                coord_pred = torch.cat([xy_pred, wh_pred], dim=1)  # [num_responsible, 4]
                coord_target = target_boxes[responsible_mask]  # [num_responsible, 4]
                coord_loss += F.mse_loss(coord_pred, coord_target)
                
                # Classification loss (only for responsible predictions)
                # pred_classes[b] shape: [A, num_classes, H, W], need [A, H, W, num_classes] then mask
                class_pred = pred_classes[b].permute(0, 2, 3, 1)[responsible_mask]  # [num_responsible, num_classes]
                class_target = target_classes[responsible_mask]  # [num_responsible, num_classes]
                class_loss += F.cross_entropy(class_pred, class_target.argmax(dim=1))
            
            # Confidence loss
            conf_target = obj_mask.float()
            conf_pred = pred_conf[b]
            
            # Object confidence loss
            obj_conf_loss = F.binary_cross_entropy(
                conf_pred[responsible_mask],
                conf_target[responsible_mask]
            ) if responsible_mask.sum() > 0 else 0
            
            # No-object confidence loss
            noobj_mask = ~responsible_mask
            noobj_conf_loss = F.binary_cross_entropy(
                conf_pred[noobj_mask],
                conf_target[noobj_mask]
            ) if noobj_mask.sum() > 0 else 0
            
            conf_loss += obj_conf_loss + self.lambda_noobj * noobj_conf_loss
        
        # Combine losses
        total_loss = (
            self.lambda_coord * coord_loss +
            conf_loss +
            class_loss
        )
        
        return total_loss, {
            'coord_loss': coord_loss,
            'conf_loss': conf_loss,
            'class_loss': class_loss,
            'total_loss': total_loss
        }
    
    def calculate_iou(self, box1, box2):
        """Calculate IoU between two boxes (w, h format)"""
        # For anchor matching, we only compare areas
        area1 = box1[0] * box1[1]
        area2 = box2[0] * box2[1]
        intersection = min(area1, area2)
        union = area1 + area2 - intersection
        return intersection / (union + 1e-6)


def create_model(num_classes=6, pretrained=False):
    """Create PCB defect detection model"""
    model = PCBDefectNet(num_classes=num_classes)
    
    if pretrained:
        # Initialize with Xavier/He initialization for better convergence
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    return model


if __name__ == "__main__":
    # Test model
    model = create_model(num_classes=6)
    
    # Test with sample input (grayscale 600x600)
    x = torch.randn(2, 1, 600, 600)
    
    with torch.no_grad():
        output = model(x)
        print(f"Model output shape: {output.shape}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")