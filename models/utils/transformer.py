import torch

def nchw_to_nlc(x):
    """Flatten [N, C, H, W] shape tensor to [N, L, C] shape tensor.
    
    Args:
        x (Tensor): The input tensor of shape [N, C, H, W] before conversion.
        
    Returns:
        Tensor: The output tensor of shape [N, L, C] after conversion.
    """
    N, C, H, W = x.shape
    x = x.flatten(2).transpose(1, 2)
    return x

def nlc_to_nchw(x, hw_shape):
    """Convert [N, L, C] shape tensor to [N, C, H, W] shape tensor.
    
    Args:
        x (Tensor): The input tensor of shape [N, L, C] before conversion.
        hw_shape (Sequence[int]): The height and width of the output feature map.
        
    Returns:
        Tensor: The output tensor of shape [N, C, H, W] after conversion.
    """
    H, W = hw_shape
    N, L, C = x.shape
    x = x.transpose(1, 2).reshape(N, C, H, W)
    return x 