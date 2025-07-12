"""Performance profiling and monitoring utilities for SPINE.

This module provides tools to profile and monitor the performance of various
components in the SPINE codebase, helping identify bottlenecks and track
optimization improvements.
"""

import time
import functools
import threading
import psutil
import numpy as np
from collections import defaultdict, deque
from contextlib import contextmanager
from typing import Dict, List, Optional, Any, Callable
import logging

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.profiler
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. Some profiling features will be disabled.")

__all__ = ['PerformanceProfiler', 'profile_function', 'profile_context', 
           'MemoryTracker', 'GPUMonitor', 'BatchSizeOptimizer']


class PerformanceProfiler:
    """Comprehensive performance profiler for SPINE operations."""
    
    def __init__(self, enabled=True, max_history=1000):
        """Initialize the performance profiler.
        
        Parameters
        ----------
        enabled : bool, default True
            Whether profiling is enabled
        max_history : int, default 1000
            Maximum number of profile records to keep
        """
        self.enabled = enabled
        self.max_history = max_history
        self.profiles = defaultdict(lambda: deque(maxlen=max_history))
        self.start_times = {}
        self.lock = threading.Lock()
        
    def start(self, name: str):
        """Start profiling an operation.
        
        Parameters
        ----------
        name : str
            Name of the operation to profile
        """
        if not self.enabled:
            return
            
        with self.lock:
            self.start_times[name] = time.perf_counter()
    
    def stop(self, name: str, metadata: Optional[Dict] = None):
        """Stop profiling an operation.
        
        Parameters
        ----------
        name : str
            Name of the operation to profile
        metadata : dict, optional
            Additional metadata to store
        """
        if not self.enabled:
            return
            
        end_time = time.perf_counter()
        with self.lock:
            start_time = self.start_times.pop(name, end_time)
            duration = end_time - start_time
            
            profile_data = {
                'duration': duration,
                'timestamp': end_time,
                'metadata': metadata or {}
            }
            
            self.profiles[name].append(profile_data)
    
    @contextmanager
    def profile(self, name: str, metadata: Optional[Dict] = None):
        """Context manager for profiling operations.
        
        Parameters
        ----------
        name : str
            Name of the operation to profile
        metadata : dict, optional
            Additional metadata to store
        """
        self.start(name)
        try:
            yield
        finally:
            self.stop(name, metadata)
    
    def get_stats(self, name: str) -> Dict[str, Any]:
        """Get performance statistics for an operation.
        
        Parameters
        ----------
        name : str
            Name of the operation
            
        Returns
        -------
        dict
            Performance statistics
        """
        if name not in self.profiles:
            return {}
        
        durations = [p['duration'] for p in self.profiles[name]]
        if not durations:
            return {}
        
        return {
            'count': len(durations),
            'mean': np.mean(durations),
            'std': np.std(durations),
            'min': np.min(durations),
            'max': np.max(durations),
            'median': np.median(durations),
            'p95': np.percentile(durations, 95),
            'p99': np.percentile(durations, 99),
            'total': np.sum(durations)
        }
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get performance statistics for all operations.
        
        Returns
        -------
        dict
            Performance statistics for all operations
        """
        return {name: self.get_stats(name) for name in self.profiles.keys()}
    
    def clear(self):
        """Clear all profiling data."""
        with self.lock:
            self.profiles.clear()
            self.start_times.clear()


class MemoryTracker:
    """Track memory usage during operations."""
    
    def __init__(self, track_gpu=True):
        """Initialize the memory tracker.
        
        Parameters
        ----------
        track_gpu : bool, default True
            Whether to track GPU memory usage
        """
        self.track_gpu = track_gpu and TORCH_AVAILABLE
        self.memory_snapshots = []
        
    def take_snapshot(self, name: str = "") -> Dict[str, Any]:
        """Take a memory usage snapshot.
        
        Parameters
        ----------
        name : str, optional
            Name for the snapshot
            
        Returns
        -------
        dict
            Memory usage information
        """
        snapshot = {
            'name': name,
            'timestamp': time.time(),
            'cpu_memory': {
                'rss': psutil.Process().memory_info().rss / 1024**2,  # MB
                'vms': psutil.Process().memory_info().vms / 1024**2,  # MB
                'percent': psutil.Process().memory_percent()
            }
        }
        
        if self.track_gpu and torch.cuda.is_available():
            snapshot['gpu_memory'] = {
                'allocated': torch.cuda.memory_allocated() / 1024**2,  # MB
                'reserved': torch.cuda.memory_reserved() / 1024**2,    # MB
                'max_allocated': torch.cuda.max_memory_allocated() / 1024**2,  # MB
                'total': torch.cuda.get_device_properties(0).total_memory / 1024**2  # MB
            }
        
        self.memory_snapshots.append(snapshot)
        return snapshot
    
    def get_memory_diff(self, start_snapshot: Dict, end_snapshot: Dict) -> Dict[str, Any]:
        """Calculate memory usage difference between snapshots.
        
        Parameters
        ----------
        start_snapshot : dict
            Starting memory snapshot
        end_snapshot : dict
            Ending memory snapshot
            
        Returns
        -------
        dict
            Memory usage difference
        """
        diff = {
            'duration': end_snapshot['timestamp'] - start_snapshot['timestamp'],
            'cpu_memory_diff': {
                'rss': end_snapshot['cpu_memory']['rss'] - start_snapshot['cpu_memory']['rss'],
                'vms': end_snapshot['cpu_memory']['vms'] - start_snapshot['cpu_memory']['vms']
            }
        }
        
        if 'gpu_memory' in start_snapshot and 'gpu_memory' in end_snapshot:
            diff['gpu_memory_diff'] = {
                'allocated': (end_snapshot['gpu_memory']['allocated'] - 
                            start_snapshot['gpu_memory']['allocated']),
                'reserved': (end_snapshot['gpu_memory']['reserved'] - 
                           start_snapshot['gpu_memory']['reserved'])
            }
        
        return diff
    
    def clear_snapshots(self):
        """Clear all memory snapshots."""
        self.memory_snapshots.clear()


class GPUMonitor:
    """Monitor GPU utilization and performance."""
    
    def __init__(self, enabled=True):
        """Initialize the GPU monitor.
        
        Parameters
        ----------
        enabled : bool, default True
            Whether GPU monitoring is enabled
        """
        self.enabled = enabled and TORCH_AVAILABLE
        self.metrics = defaultdict(list)
        
    def record_gpu_stats(self, name: str = "default"):
        """Record current GPU statistics.
        
        Parameters
        ----------
        name : str, default "default"
            Name for the metric group
        """
        if not self.enabled or not torch.cuda.is_available():
            return
        
        stats = {
            'timestamp': time.time(),
            'memory_allocated': torch.cuda.memory_allocated() / 1024**2,
            'memory_reserved': torch.cuda.memory_reserved() / 1024**2,
            'memory_free': (torch.cuda.get_device_properties(0).total_memory - 
                          torch.cuda.memory_allocated()) / 1024**2
        }
        
        self.metrics[name].append(stats)
    
    def get_gpu_utilization_summary(self, name: str = "default") -> Dict[str, Any]:
        """Get GPU utilization summary.
        
        Parameters
        ----------
        name : str, default "default"
            Name for the metric group
            
        Returns
        -------
        dict
            GPU utilization summary
        """
        if name not in self.metrics:
            return {}
        
        stats = self.metrics[name]
        if not stats:
            return {}
        
        memory_allocated = [s['memory_allocated'] for s in stats]
        memory_free = [s['memory_free'] for s in stats]
        
        return {
            'memory_allocated': {
                'mean': np.mean(memory_allocated),
                'max': np.max(memory_allocated),
                'min': np.min(memory_allocated)
            },
            'memory_free': {
                'mean': np.mean(memory_free),
                'max': np.max(memory_free),
                'min': np.min(memory_free)
            },
            'num_measurements': len(stats)
        }


class BatchSizeOptimizer:
    """Optimize batch size based on performance metrics."""
    
    def __init__(self, initial_batch_size=32, max_batch_size=512, 
                 min_batch_size=1, adaptation_factor=1.1):
        """Initialize the batch size optimizer.
        
        Parameters
        ----------
        initial_batch_size : int, default 32
            Initial batch size
        max_batch_size : int, default 512
            Maximum allowed batch size
        min_batch_size : int, default 1
            Minimum allowed batch size
        adaptation_factor : float, default 1.1
            Factor for increasing/decreasing batch size
        """
        self.current_batch_size = initial_batch_size
        self.max_batch_size = max_batch_size
        self.min_batch_size = min_batch_size
        self.adaptation_factor = adaptation_factor
        self.performance_history = deque(maxlen=10)
        
    def record_performance(self, batch_size: int, throughput: float, 
                         memory_usage: float, oom_occurred: bool = False):
        """Record performance metrics for a batch size.
        
        Parameters
        ----------
        batch_size : int
            Batch size used
        throughput : float
            Throughput (samples/second)
        memory_usage : float
            Memory usage (MB)
        oom_occurred : bool, default False
            Whether out-of-memory occurred
        """
        self.performance_history.append({
            'batch_size': batch_size,
            'throughput': throughput,
            'memory_usage': memory_usage,
            'oom_occurred': oom_occurred,
            'timestamp': time.time()
        })
        
    def suggest_batch_size(self) -> int:
        """Suggest optimal batch size based on performance history.
        
        Returns
        -------
        int
            Suggested batch size
        """
        if len(self.performance_history) < 2:
            return self.current_batch_size
        
        # Get recent performance metrics
        recent_metrics = list(self.performance_history)[-3:]
        
        # If OOM occurred, reduce batch size
        if any(m['oom_occurred'] for m in recent_metrics):
            new_batch_size = max(self.min_batch_size, 
                                int(self.current_batch_size / self.adaptation_factor))
        else:
            # Check if throughput is improving with larger batch sizes
            throughputs = [m['throughput'] for m in recent_metrics]
            if len(throughputs) >= 2 and throughputs[-1] > throughputs[-2]:
                # Increase batch size
                new_batch_size = min(self.max_batch_size, 
                                   int(self.current_batch_size * self.adaptation_factor))
            else:
                # Keep current batch size
                new_batch_size = self.current_batch_size
        
        self.current_batch_size = new_batch_size
        return new_batch_size


def profile_function(name: str = None, profiler: PerformanceProfiler = None):
    """Decorator to profile function execution time.
    
    Parameters
    ----------
    name : str, optional
        Name for the profile (uses function name if not provided)
    profiler : PerformanceProfiler, optional
        Profiler instance to use
        
    Returns
    -------
    callable
        Decorated function
    """
    def decorator(func: Callable):
        nonlocal profiler
        if profiler is None:
            profiler = PerformanceProfiler()
        
        profile_name = name or func.__name__
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with profiler.profile(profile_name):
                return func(*args, **kwargs)
        
        wrapper._profiler = profiler
        return wrapper
    
    return decorator


@contextmanager
def profile_context(name: str, profiler: PerformanceProfiler = None):
    """Context manager for profiling code blocks.
    
    Parameters
    ----------
    name : str
        Name for the profile
    profiler : PerformanceProfiler, optional
        Profiler instance to use
    """
    if profiler is None:
        profiler = PerformanceProfiler()
    
    with profiler.profile(name):
        yield profiler


# Global profiler instance
_global_profiler = PerformanceProfiler()
_global_memory_tracker = MemoryTracker()
_global_gpu_monitor = GPUMonitor()


def get_global_profiler() -> PerformanceProfiler:
    """Get the global profiler instance."""
    return _global_profiler


def get_global_memory_tracker() -> MemoryTracker:
    """Get the global memory tracker instance."""
    return _global_memory_tracker


def get_global_gpu_monitor() -> GPUMonitor:
    """Get the global GPU monitor instance."""
    return _global_gpu_monitor


def print_performance_summary():
    """Print a summary of all performance metrics."""
    print("\n=== SPINE Performance Summary ===")
    
    # Profiler stats
    stats = _global_profiler.get_all_stats()
    if stats:
        print("\nFunction Performance:")
        for name, data in stats.items():
            print(f"  {name}:")
            print(f"    Count: {data['count']}")
            print(f"    Mean: {data['mean']:.4f}s")
            print(f"    Total: {data['total']:.4f}s")
            print(f"    P95: {data['p95']:.4f}s")
    
    # GPU utilization
    gpu_stats = _global_gpu_monitor.get_gpu_utilization_summary()
    if gpu_stats:
        print("\nGPU Utilization:")
        print(f"  Memory Allocated (avg): {gpu_stats['memory_allocated']['mean']:.2f} MB")
        print(f"  Memory Free (avg): {gpu_stats['memory_free']['mean']:.2f} MB")
    
    print("\n" + "="*35)