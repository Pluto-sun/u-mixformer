import torch
import torch.nn as nn
from ..backbones.mit import MixVisionTransformer

class UMixFormerClassifier(nn.Module):
    """UMixFormer for Image Classification.
    
    This classifier is based on the UMixFormer architecture, which uses
    a hierarchical vision transformer as backbone and adds a classification head.
    
    Args:
        backbone_config (dict): Backbone configuration dictionary.
        num_classes (int): Number of classes for classification.
        in_channels (int): Number of input channels. Defaults to 3.
        drop_rate (float): Dropout rate. Defaults to 0.0.
        drop_path_rate (float): Drop path rate. Defaults to 0.1.
    """
    
    def __init__(self,
                 backbone_config,
                 num_classes,
                 in_channels=3,
                 drop_rate=0.0,
                 drop_path_rate=0.1):
        super().__init__()
        
        # Build backbone
        self.backbone = MixVisionTransformer(**backbone_config)
        
        # Get the output dimension of the backbone
        self.embed_dims = self.backbone.embed_dims[-1]
        
        # Classification head
        self.head = nn.Sequential(
            nn.LayerNorm(self.embed_dims),
            nn.Linear(self.embed_dims, num_classes)
        )
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Dropout
        self.dropout = nn.Dropout(p=drop_rate)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize the weights of the model."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        
    def forward(self, x):
        """Forward function.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).
            
        Returns:
            torch.Tensor: Classification logits of shape (B, num_classes).
        """
        # Get features from backbone
        features = self.backbone(x)
        
        # Get the last feature map
        x = features[-1]  # (B, C, H, W)
        
        # Global average pooling
        x = self.avgpool(x)  # (B, C, 1, 1)
        x = torch.flatten(x, 1)  # (B, C)
        
        # Classification head
        x = self.dropout(x)
        x = self.head(x)  # (B, num_classes)
        
        return x
    
    def extract_features(self, x):
        """Extract features from the backbone.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).
            
        Returns:
            list[torch.Tensor]: List of feature maps from different stages.
        """
        return self.backbone(x) 