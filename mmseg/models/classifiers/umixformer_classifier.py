import torch
import torch.nn as nn
from mmengine.model import BaseModule
from mmseg.registry import MODELS
from ..backbones.mit import MixVisionTransformer
from ..utils import nchw_to_nlc, nlc_to_nchw

@MODELS.register_module()
class UMixFormerClassifier(BaseModule):
    """UMixFormer for Image Classification.
    
    This classifier is based on the UMixFormer architecture, which uses
    a hierarchical vision transformer as backbone and adds a classification head.
    
    Args:
        backbone (dict): Backbone config dict.
        num_classes (int): Number of classes for classification.
        in_channels (int): Number of input channels. Defaults to 3.
        drop_rate (float): Dropout rate. Defaults to 0.0.
        drop_path_rate (float): Drop path rate. Defaults to 0.1.
        init_cfg (dict, optional): Initialization config dict. Defaults to None.
    """
    
    def __init__(self,
                 backbone,
                 num_classes,
                 in_channels=3,
                 drop_rate=0.0,
                 drop_path_rate=0.1,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        
        # Build backbone
        self.backbone = MODELS.build(backbone)
        
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