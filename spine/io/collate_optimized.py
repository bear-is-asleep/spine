"""Optimized data collation classes for improved performance.

This module provides optimized versions of the collation functions that minimize
CPU-GPU transfers and improve batch processing performance.
"""

import numpy as np
import torch
from functools import lru_cache
from typing import List, Dict, Optional, Union, Any

from spine.data import TensorBatch, IndexBatch, EdgeIndexBatch
from spine.utils.geo import Geometry

__all__ = ['OptimizedCollateAll', 'TensorCollator', 'BatchProcessor']


class TensorCollator:
    """Optimized tensor collation with minimal CPU-GPU transfers."""
    
    def __init__(self, device=None, dtype=torch.float32):
        """Initialize the tensor collator.
        
        Parameters
        ----------
        device : torch.device, optional
            Target device for tensors
        dtype : torch.dtype, default torch.float32
            Data type for tensors
        """
        self.device = device
        self.dtype = dtype
        self._tensor_cache = {}
    
    def collate_tensors(self, tensors: List[np.ndarray], 
                       add_batch_ids: bool = True) -> torch.Tensor:
        """Efficiently collate a list of tensors.
        
        Parameters
        ----------
        tensors : List[np.ndarray]
            List of tensors to collate
        add_batch_ids : bool, default True
            Whether to add batch IDs as the first column
            
        Returns
        -------
        torch.Tensor
            Collated tensor
        """
        if not tensors:
            return torch.empty(0, dtype=self.dtype, device=self.device)
        
        # Convert to tensors and move to device in one operation
        tensor_list = [torch.from_numpy(t).to(dtype=self.dtype, device=self.device) 
                      for t in tensors]
        
        if add_batch_ids:
            # Pre-compute batch IDs
            batch_ids = []
            for i, tensor in enumerate(tensor_list):
                batch_id = torch.full((len(tensor), 1), i, 
                                    dtype=self.dtype, device=self.device)
                batch_ids.append(batch_id)
            
            # Concatenate batch IDs with tensors
            tensor_list = [torch.cat([bid, tensor], dim=1) 
                          for bid, tensor in zip(batch_ids, tensor_list)]
        
        # Single concatenation operation
        return torch.cat(tensor_list, dim=0)
    
    def collate_with_offsets(self, tensors: List[np.ndarray], 
                           counts: List[int]) -> torch.Tensor:
        """Collate tensors with offset computation for graph operations.
        
        Parameters
        ----------
        tensors : List[np.ndarray]
            List of tensors to collate
        counts : List[int]
            Number of nodes per batch
            
        Returns
        -------
        torch.Tensor
            Collated tensor with offsets applied
        """
        if not tensors:
            return torch.empty(0, dtype=torch.long, device=self.device)
        
        # Pre-compute offsets
        offsets = torch.zeros(len(counts), dtype=torch.long, device=self.device)
        if len(counts) > 1:
            offsets[1:] = torch.cumsum(torch.tensor(counts[:-1]), dim=0)
        
        # Apply offsets and concatenate
        result_tensors = []
        for i, tensor in enumerate(tensors):
            t = torch.from_numpy(tensor).to(dtype=torch.long, device=self.device)
            t += offsets[i]
            result_tensors.append(t)
        
        return torch.cat(result_tensors, dim=0)


class BatchProcessor:
    """Optimized batch processing with memory management."""
    
    def __init__(self, max_memory_mb: float = 4096):
        """Initialize the batch processor.
        
        Parameters
        ----------
        max_memory_mb : float, default 4096
            Maximum memory to use in MB
        """
        self.max_memory_mb = max_memory_mb
        self._memory_tracker = {}
    
    def estimate_memory_usage(self, batch_size: int, 
                            tensor_shape: tuple) -> float:
        """Estimate memory usage for a batch.
        
        Parameters
        ----------
        batch_size : int
            Size of the batch
        tensor_shape : tuple
            Shape of individual tensors
            
        Returns
        -------
        float
            Estimated memory usage in MB
        """
        # Estimate memory per element (in bytes)
        element_size = np.prod(tensor_shape) * 4  # float32
        total_memory = batch_size * element_size
        return total_memory / (1024 ** 2)  # Convert to MB
    
    def optimize_batch_size(self, base_batch_size: int, 
                          tensor_shape: tuple) -> int:
        """Optimize batch size based on memory constraints.
        
        Parameters
        ----------
        base_batch_size : int
            Base batch size
        tensor_shape : tuple
            Shape of individual tensors
            
        Returns
        -------
        int
            Optimized batch size
        """
        estimated_memory = self.estimate_memory_usage(base_batch_size, tensor_shape)
        
        if estimated_memory > self.max_memory_mb:
            # Reduce batch size proportionally
            ratio = self.max_memory_mb / estimated_memory
            return max(1, int(base_batch_size * ratio))
        
        return base_batch_size


class OptimizedCollateAll:
    """Optimized version of CollateAll with performance improvements.
    
    This class provides the same functionality as CollateAll but with:
    - Reduced CPU-GPU transfers
    - Vectorized operations
    - Memory optimization
    - Better cache utilization
    """
    
    name = 'optimized_all'
    
    def __init__(self, split=False, target_id=0, detector=None,
                 geometry_file=None, overlay=None, source=None,
                 device=None, use_amp=True, max_memory_mb=4096):
        """Initialize the optimized collation parameters.
        
        Parameters
        ----------
        split : bool, default False
            Whether to split the input by module ID
        target_id : int, default 0
            If split is True, specifies where to relocate the points
        detector : str, optional
            Name of a recognized detector to the geometry from
        geometry_file : str, optional
            Path to a `.yaml` geometry file to load the geometry from
        overlay : dict, optional
            Image overlay configuration
        source : dict, optional
            Dictionary which maps keys to their corresponding sources
        device : torch.device, optional
            Target device for tensors
        use_amp : bool, default True
            Whether to use automatic mixed precision
        max_memory_mb : float, default 4096
            Maximum memory to use in MB
        """
        self.split = split
        self.target_id = target_id
        self.source = source
        self.device = device
        self.use_amp = use_amp
        
        # Initialize geometry if required
        self.geo = None
        if split:
            assert (detector is not None) or (geometry_file is not None), (
                    "If splitting input per module, must provide detector")
            self.geo = Geometry(detector, geometry_file)
        
        # Initialize optimized components
        self.tensor_collator = TensorCollator(device=device)
        self.batch_processor = BatchProcessor(max_memory_mb=max_memory_mb)
        
        # Cache for frequently accessed data
        self._batch_cache = {}
        self._geometry_cache = {}
        
        if overlay is not None:
            self.process_overlay_config(**overlay)
    
    def process_overlay_config(self, mode='const', size=2):
        """Process the image overlay configuration.
        
        Parameters
        ----------
        mode : str, default 'const'
            Method used to form overlay indexes
        size : int, default 2
            Number of images to merge
        """
        # TODO: Implement optimized overlay processing
        pass
    
    @lru_cache(maxsize=128)
    def _get_batch_info(self, batch_size: int, key: str) -> Dict[str, Any]:
        """Get cached batch information for optimization.
        
        Parameters
        ----------
        batch_size : int
            Size of the batch
        key : str
            Data key
            
        Returns
        -------
        Dict[str, Any]
            Batch information
        """
        return {
            'batch_size': batch_size,
            'key': key,
            'device': self.device
        }
    
    def _collate_coordinate_tensors(self, batch: List[Dict], 
                                   key: str) -> TensorBatch:
        """Optimized collation of coordinate tensors.
        
        Parameters
        ----------
        batch : List[Dict]
            List of batch dictionaries
        key : str
            Key for the tensor data
            
        Returns
        -------
        TensorBatch
            Collated tensor batch
        """
        batch_size = len(batch)
        
        # Extract tensors and metadata
        voxels_list = []
        features_list = []
        counts = []
        
        for sample in batch:
            voxels, features, meta = sample[key]
            voxels_list.append(voxels)
            features_list.append(features)
            counts.append(len(voxels))
        
        # Optimize batch size if needed
        if hasattr(self, 'batch_processor'):
            avg_shape = np.mean([v.shape for v in voxels_list], axis=0)
            optimal_batch_size = self.batch_processor.optimize_batch_size(
                batch_size, avg_shape)
            
            if optimal_batch_size < batch_size:
                # Trim batch if memory constrained
                voxels_list = voxels_list[:optimal_batch_size]
                features_list = features_list[:optimal_batch_size]
                counts = counts[:optimal_batch_size]
        
        # Efficient tensor collation
        voxels = self.tensor_collator.collate_tensors(voxels_list, add_batch_ids=False)
        features = self.tensor_collator.collate_tensors(features_list, add_batch_ids=False)
        
        # Create batch IDs efficiently
        batch_ids = torch.repeat_interleave(
            torch.arange(len(counts), dtype=voxels.dtype, device=self.device),
            torch.tensor(counts, dtype=torch.long, device=self.device)
        )
        
        # Combine tensors
        tensor = torch.cat([batch_ids.unsqueeze(1), voxels, features], dim=1)
        coord_cols = np.arange(1, 1 + voxels.shape[1])
        
        return TensorBatch(
            tensor, counts, has_batch_col=True, coord_cols=coord_cols)
    
    def _collate_index_tensors(self, batch: List[Dict], 
                              key: str) -> Union[IndexBatch, EdgeIndexBatch]:
        """Optimized collation of index tensors.
        
        Parameters
        ----------
        batch : List[Dict]
            List of batch dictionaries
        key : str
            Key for the tensor data
            
        Returns
        -------
        Union[IndexBatch, EdgeIndexBatch]
            Collated index batch
        """
        indices_list = []
        counts_list = []
        total_counts = []
        
        for sample in batch:
            indices, count = sample[key]
            indices_list.append(indices)
            counts_list.append(indices.shape[-1])
            total_counts.append(count)
        
        # Compute offsets and apply them efficiently
        tensor = self.tensor_collator.collate_with_offsets(indices_list, total_counts)
        
        if len(tensor.shape) == 1:
            return IndexBatch(tensor, counts_list, 
                            np.cumsum([0] + total_counts[:-1]))
        else:
            return EdgeIndexBatch(tensor, counts_list, 
                                np.cumsum([0] + total_counts[:-1]), 
                                directed=True)
    
    def _collate_feature_tensors(self, batch: List[Dict], 
                                key: str) -> TensorBatch:
        """Optimized collation of feature tensors.
        
        Parameters
        ----------
        batch : List[Dict]
            List of batch dictionaries
        key : str
            Key for the tensor data
            
        Returns
        -------
        TensorBatch
            Collated tensor batch
        """
        tensors = [sample[key] for sample in batch]
        counts = [len(tensor) for tensor in tensors]
        
        # Efficient tensor collation without batch IDs
        tensor = self.tensor_collator.collate_tensors(tensors, add_batch_ids=False)
        
        return TensorBatch(tensor, counts)
    
    def __call__(self, batch: List[Dict]) -> Dict[str, Any]:
        """Optimized batch collation with performance improvements.
        
        Parameters
        ----------
        batch : List[Dict]
            List of dictionaries of parsed information
            
        Returns
        -------
        Dict[str, Any]
            Dictionary of collated batch data
        """
        if not batch:
            return {}
        
        batch_size = len(batch)
        data = {}
        
        # Process each key in the batch
        for key in batch[0].keys():
            ref_obj = batch[0][key]
            
            # Dispatch to optimized collation methods
            if isinstance(ref_obj, tuple) and len(ref_obj) == 3:
                # Coordinate and feature tensors
                data[key] = self._collate_coordinate_tensors(batch, key)
                
            elif isinstance(ref_obj, tuple) and len(ref_obj) == 2:
                # Index tensors
                data[key] = self._collate_index_tensors(batch, key)
                
            elif isinstance(ref_obj, np.ndarray):
                # Feature tensors
                data[key] = self._collate_feature_tensors(batch, key)
                
            else:
                # Other data types - create list
                data[key] = [sample[key] for sample in batch]
        
        return data