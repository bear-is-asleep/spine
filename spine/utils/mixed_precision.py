"""Mixed precision training utilities for SPINE.

This module provides utilities for enabling automatic mixed precision (AMP) training
to improve performance and reduce memory usage.
"""

import torch
from torch.cuda.amp import autocast, GradScaler
from functools import wraps
import logging

logger = logging.getLogger(__name__)

__all__ = ['AMPManager', 'autocast_if_enabled', 'mixed_precision_forward']


class AMPManager:
    """Manages automatic mixed precision training and inference.
    
    This class provides a centralized way to enable/disable mixed precision
    training and manage the gradient scaler for stable training.
    """
    
    def __init__(self, enabled=True, init_scale=2**16, growth_factor=2.0, 
                 backoff_factor=0.5, growth_interval=2000):
        """Initialize the AMP manager.
        
        Parameters
        ----------
        enabled : bool, default True
            Whether to enable automatic mixed precision
        init_scale : float, default 2**16
            Initial scale value for gradient scaler
        growth_factor : float, default 2.0
            Factor by which to multiply scale on successful steps
        backoff_factor : float, default 0.5
            Factor by which to multiply scale on overflow
        growth_interval : int, default 2000
            Number of successful steps before increasing scale
        """
        self.enabled = enabled and torch.cuda.is_available()
        
        if self.enabled:
            self.scaler = GradScaler(
                init_scale=init_scale,
                growth_factor=growth_factor,
                backoff_factor=backoff_factor,
                growth_interval=growth_interval
            )
            logger.info("Mixed precision training enabled")
        else:
            self.scaler = None
            logger.info("Mixed precision training disabled")
    
    def scale_loss(self, loss):
        """Scale the loss for mixed precision training.
        
        Parameters
        ----------
        loss : torch.Tensor
            The loss tensor to scale
            
        Returns
        -------
        torch.Tensor
            Scaled loss tensor
        """
        if self.enabled:
            return self.scaler.scale(loss)
        return loss
    
    def step(self, optimizer):
        """Perform optimizer step with gradient scaling.
        
        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            The optimizer to step
            
        Returns
        -------
        bool
            Whether the step was successful (no overflow)
        """
        if self.enabled:
            self.scaler.step(optimizer)
            self.scaler.update()
            return True
        else:
            optimizer.step()
            return True
    
    def unscale_gradients(self, optimizer):
        """Unscale gradients before clipping.
        
        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            The optimizer containing gradients to unscale
        """
        if self.enabled:
            self.scaler.unscale_(optimizer)
    
    def get_scale(self):
        """Get the current gradient scale.
        
        Returns
        -------
        float
            Current gradient scale value
        """
        if self.enabled:
            return self.scaler.get_scale()
        return 1.0


def autocast_if_enabled(enabled=True):
    """Decorator to conditionally enable autocast for model forward passes.
    
    Parameters
    ----------
    enabled : bool, default True
        Whether to enable autocast
        
    Returns
    -------
    callable
        Decorated function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if enabled and torch.cuda.is_available():
                with autocast():
                    return func(*args, **kwargs)
            else:
                return func(*args, **kwargs)
        return wrapper
    return decorator


def mixed_precision_forward(model, *args, use_amp=True, **kwargs):
    """Execute model forward pass with optional mixed precision.
    
    Parameters
    ----------
    model : torch.nn.Module
        The model to execute
    *args : tuple
        Positional arguments for the model
    use_amp : bool, default True
        Whether to use automatic mixed precision
    **kwargs : dict
        Keyword arguments for the model
        
    Returns
    -------
    Any
        Model output
    """
    if use_amp and torch.cuda.is_available():
        with autocast():
            return model(*args, **kwargs)
    else:
        return model(*args, **kwargs)


class MemoryOptimizer:
    """Utilities for optimizing GPU memory usage."""
    
    @staticmethod
    def clear_cache():
        """Clear GPU memory cache."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    @staticmethod
    def get_memory_info():
        """Get current GPU memory usage information.
        
        Returns
        -------
        dict
            Memory usage information
        """
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated()
            cached = torch.cuda.memory_reserved()
            max_allocated = torch.cuda.max_memory_allocated()
            
            return {
                'allocated_mb': allocated / 1024**2,
                'cached_mb': cached / 1024**2,
                'max_allocated_mb': max_allocated / 1024**2,
                'free_mb': (torch.cuda.get_device_properties(0).total_memory - allocated) / 1024**2
            }
        return {'allocated_mb': 0, 'cached_mb': 0, 'max_allocated_mb': 0, 'free_mb': 0}
    
    @staticmethod
    def optimize_batch_size(base_batch_size, max_memory_mb=None):
        """Optimize batch size based on available GPU memory.
        
        Parameters
        ----------
        base_batch_size : int
            Base batch size to start with
        max_memory_mb : float, optional
            Maximum memory to use in MB
            
        Returns
        -------
        int
            Optimized batch size
        """
        if not torch.cuda.is_available():
            return base_batch_size
        
        memory_info = MemoryOptimizer.get_memory_info()
        available_memory = memory_info['free_mb']
        
        if max_memory_mb is None:
            # Use 80% of available memory
            max_memory_mb = available_memory * 0.8
        
        # Estimate memory per sample (rough heuristic)
        memory_per_sample = max_memory_mb / base_batch_size
        optimal_batch_size = int(max_memory_mb / memory_per_sample)
        
        return max(1, min(optimal_batch_size, base_batch_size))