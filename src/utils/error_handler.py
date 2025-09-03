"""Robust error handling for memory-intensive AI model operations."""

import gc
import traceback
import functools
import time
from typing import Any, Callable, Dict, List, Optional, Type, Union
from contextlib import contextmanager
from .logger import get_logger
from .memory_manager import memory_manager
from .system_optimizer import system_optimizer

logger = get_logger(__name__)

class MemoryError(Exception):
    """Custom exception for memory-related errors."""
    pass

class ModelLoadError(Exception):
    """Custom exception for model loading errors."""
    pass

class InferenceError(Exception):
    """Custom exception for model inference errors."""
    pass

class ErrorHandler:
    """Handles errors in memory-intensive AI operations with recovery strategies."""
    
    def __init__(self):
        self.error_counts = {}
        self.recovery_attempts = {}
        self.max_retries = 3
        self.retry_delay = 2.0
        
    def handle_memory_error(self, error: Exception, operation: str, **kwargs) -> bool:
        """Handle memory-related errors with recovery strategies."""
        error_key = f"{operation}_{type(error).__name__}"
        
        # Track error frequency
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        
        logger.error(f"Memory error in {operation}: {error}")
        logger.info(f"Error count for {error_key}: {self.error_counts[error_key]}")
        
        # Apply recovery strategies based on error type and frequency
        if self.error_counts[error_key] <= self.max_retries:
            return self._apply_memory_recovery_strategy(error, operation, **kwargs)
        else:
            logger.error(f"Max retries exceeded for {error_key}")
            return False
    
    def _apply_memory_recovery_strategy(self, error: Exception, operation: str, **kwargs) -> bool:
        """Apply appropriate recovery strategy based on error type."""
        error_type = type(error).__name__
        
        logger.info(f"Applying recovery strategy for {error_type} in {operation}")
        
        try:
            if 'cuda' in str(error).lower() or 'gpu' in str(error).lower():
                return self._handle_gpu_memory_error(error, operation, **kwargs)
            elif 'memory' in str(error).lower() or isinstance(error, MemoryError):
                return self._handle_system_memory_error(error, operation, **kwargs)
            elif 'out of memory' in str(error).lower():
                return self._handle_oom_error(error, operation, **kwargs)
            else:
                return self._handle_generic_error(error, operation, **kwargs)
                
        except Exception as recovery_error:
            logger.error(f"Recovery strategy failed: {recovery_error}")
            return False
    
    def _handle_gpu_memory_error(self, error: Exception, operation: str, **kwargs) -> bool:
        """Handle GPU memory errors."""
        logger.info("Handling GPU memory error...")
        
        # Clear GPU cache
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                logger.info("Cleared GPU cache")
        except ImportError:
            pass
        
        # Force CPU fallback
        if 'force_cpu' not in kwargs or not kwargs['force_cpu']:
            logger.info("Forcing CPU fallback for operation")
            kwargs['force_cpu'] = True
            kwargs['device'] = 'cpu'
            return True
        
        return False
    
    def _handle_system_memory_error(self, error: Exception, operation: str, **kwargs) -> bool:
        """Handle system memory errors."""
        logger.info("Handling system memory error...")
        
        # Aggressive memory cleanup
        memory_manager.aggressive_cleanup()
        
        # Force garbage collection multiple times
        for i in range(3):
            collected = gc.collect()
            logger.info(f"GC pass {i+1}: collected {collected} objects")
            time.sleep(0.5)
        
        # Check if memory is now available
        memory_status = memory_manager.get_memory_status()
        if memory_status['available_gb'] > 1.0:  # At least 1GB available
            logger.info(f"Memory recovered: {memory_status['available_gb']:.1f}GB available")
            return True
        
        # Try with reduced batch size or model size
        if 'batch_size' in kwargs and kwargs['batch_size'] > 1:
            kwargs['batch_size'] = max(1, kwargs['batch_size'] // 2)
            logger.info(f"Reduced batch size to {kwargs['batch_size']}")
            return True
        
        return False
    
    def _handle_oom_error(self, error: Exception, operation: str, **kwargs) -> bool:
        """Handle out-of-memory errors."""
        logger.info("Handling out-of-memory error...")
        
        # Combine GPU and system memory recovery
        gpu_recovery = self._handle_gpu_memory_error(error, operation, **kwargs)
        system_recovery = self._handle_system_memory_error(error, operation, **kwargs)
        
        return gpu_recovery or system_recovery
    
    def _handle_generic_error(self, error: Exception, operation: str, **kwargs) -> bool:
        """Handle generic errors with basic recovery."""
        logger.info("Handling generic error with basic recovery...")
        
        # Basic cleanup
        gc.collect()
        
        # Wait a bit for system to stabilize
        time.sleep(self.retry_delay)
        
        return True
    
    @contextmanager
    def error_recovery_context(self, operation: str, **recovery_kwargs):
        """Context manager for automatic error recovery."""
        try:
            yield
        except Exception as e:
            logger.error(f"Error in {operation}: {e}")
            
            # Try to recover
            if self.handle_memory_error(e, operation, **recovery_kwargs):
                logger.info(f"Recovery successful for {operation}, retrying...")
                # Caller should retry the operation
                raise RecoveryAttemptException(f"Recovery attempted for {operation}", original_error=e)
            else:
                logger.error(f"Recovery failed for {operation}")
                raise
    
    def retry_with_recovery(self, max_attempts: int = 3, delay: float = 2.0):
        """Decorator for automatic retry with error recovery."""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                operation = f"{func.__module__}.{func.__name__}"
                
                for attempt in range(max_attempts):
                    try:
                        return func(*args, **kwargs)
                    
                    except RecoveryAttemptException as recovery_e:
                        if attempt < max_attempts - 1:
                            logger.info(f"Retry attempt {attempt + 1}/{max_attempts} for {operation}")
                            time.sleep(delay * (attempt + 1))  # Exponential backoff
                            continue
                        else:
                            # Re-raise original error on final attempt
                            raise recovery_e.original_error
                    
                    except Exception as e:
                        if attempt < max_attempts - 1:
                            # Try recovery
                            recovery_kwargs = {
                                'attempt': attempt,
                                'max_attempts': max_attempts
                            }
                            
                            if self.handle_memory_error(e, operation, **recovery_kwargs):
                                logger.info(f"Recovery successful, retry attempt {attempt + 1}/{max_attempts}")
                                time.sleep(delay * (attempt + 1))
                                continue
                        
                        # If recovery failed or final attempt, re-raise
                        raise
                
                # Should not reach here
                raise RuntimeError(f"Unexpected end of retry loop for {operation}")
            
            return wrapper
        return decorator
    
    def safe_model_operation(self, operation_func: Callable, operation_name: str, **kwargs) -> Any:
        """Safely execute a model operation with error handling and recovery."""
        recovery_kwargs = kwargs.copy()
        
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Executing {operation_name} (attempt {attempt + 1}/{self.max_retries})")
                
                # Monitor memory before operation
                memory_before = memory_manager.get_memory_status()
                logger.debug(f"Memory before {operation_name}: {memory_before}")
                
                # Execute operation
                result = operation_func(**recovery_kwargs)
                
                # Monitor memory after operation
                memory_after = memory_manager.get_memory_status()
                logger.debug(f"Memory after {operation_name}: {memory_after}")
                
                return result
                
            except Exception as e:
                logger.error(f"Error in {operation_name} (attempt {attempt + 1}): {e}")
                
                if attempt < self.max_retries - 1:
                    # Try recovery
                    if self.handle_memory_error(e, operation_name, **recovery_kwargs):
                        logger.info(f"Recovery successful for {operation_name}, retrying...")
                        time.sleep(self.retry_delay * (attempt + 1))
                        continue
                
                # Final attempt or recovery failed
                logger.error(f"Failed to execute {operation_name} after {attempt + 1} attempts")
                raise
        
        raise RuntimeError(f"Unexpected end of retry loop for {operation_name}")
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics and recovery information."""
        return {
            'error_counts': self.error_counts.copy(),
            'recovery_attempts': self.recovery_attempts.copy(),
            'total_errors': sum(self.error_counts.values()),
            'max_retries': self.max_retries,
            'retry_delay': self.retry_delay
        }
    
    def reset_statistics(self) -> None:
        """Reset error statistics."""
        self.error_counts.clear()
        self.recovery_attempts.clear()
        logger.info("Error statistics reset")

class RecoveryAttemptException(Exception):
    """Exception raised when a recovery attempt is made."""
    
    def __init__(self, message: str, original_error: Exception):
        super().__init__(message)
        self.original_error = original_error

# Specific error handlers for different types of operations

def handle_model_loading_error(func: Callable) -> Callable:
    """Decorator for handling model loading errors."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Model loading error in {func.__name__}: {e}")
            
            # Try CPU fallback if GPU error
            if 'cuda' in str(e).lower() or 'gpu' in str(e).lower():
                logger.info("Attempting CPU fallback for model loading")
                kwargs['device'] = 'cpu'
                kwargs['force_cpu'] = True
                
                try:
                    return func(*args, **kwargs)
                except Exception as cpu_error:
                    logger.error(f"CPU fallback also failed: {cpu_error}")
                    raise ModelLoadError(f"Model loading failed on both GPU and CPU: {e}") from e
            
            raise ModelLoadError(f"Model loading failed: {e}") from e
    
    return wrapper

def handle_inference_error(func: Callable) -> Callable:
    """Decorator for handling inference errors."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Inference error in {func.__name__}: {e}")
            
            # Try with reduced batch size
            if 'batch_size' in kwargs and kwargs['batch_size'] > 1:
                original_batch_size = kwargs['batch_size']
                kwargs['batch_size'] = max(1, kwargs['batch_size'] // 2)
                logger.info(f"Retrying with reduced batch size: {original_batch_size} -> {kwargs['batch_size']}")
                
                try:
                    return func(*args, **kwargs)
                except Exception as batch_error:
                    logger.error(f"Reduced batch size also failed: {batch_error}")
            
            raise InferenceError(f"Inference failed: {e}") from e
    
    return wrapper

# Global error handler instance
error_handler = ErrorHandler()