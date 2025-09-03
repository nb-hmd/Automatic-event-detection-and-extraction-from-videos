from typing import List, Dict, Optional
from pathlib import Path
import os
import re
from ..utils.logger import get_logger
from ..pipeline.phase1_mvp import Phase1MVP
from ..pipeline.phase2_reranker import Phase2Reranker
from .clip_extractor import ClipExtractor
from ..utils.config import settings
from ..utils.memory_manager import memory_manager
from ..utils.progressive_loader import progressive_loader
from ..utils.system_optimizer import system_optimizer

logger = get_logger(__name__)

class VideoProcessor:
    def __init__(self, lazy_load=False, heavy_models=True):
        self.lazy_load = False  # Force disable lazy loading for heavy models
        self.heavy_models = heavy_models
        self.phase1 = None
        self.phase2 = None
        self.phase2_available = False
        self.clip_extractor = None
        self._models_loaded = False
        
        # Always load models immediately for heavy implementation - no fallbacks
        logger.info("VideoProcessor initializing with HEAVY MODEL IMPLEMENTATION - loading ALL models without fallbacks")
        self._load_models_heavy_no_fallback()
    
    def _load_models_heavy_no_fallback(self):
        """Force load ALL heavy models without any fallbacks - maximum accuracy mode."""
        if self._models_loaded:
            return
            
        logger.info("🚀 HEAVY MODEL LOADING: Loading OpenCLIP, BLIP-2, and UniVTG without fallbacks")
        
        # Log initial memory usage
        memory_manager.log_memory_usage("Before heavy model loading")
        
        try:
            # Force load Phase 1 (OpenCLIP) - no lazy loading
            logger.info("Loading Phase 1 (OpenCLIP) - HEAVY MODE")
            self.phase1 = Phase1MVP()
            logger.info("✅ Phase 1 (OpenCLIP) loaded successfully")
            
            # Force load Phase 2 (BLIP-2) - no lazy loading, no fallbacks
            logger.info("Loading Phase 2 (BLIP-2) - HEAVY MODE - NO FALLBACKS")
            self.phase2 = Phase2Reranker(lazy_load=False)  # Force immediate loading
            
            # Verify BLIP model is actually loaded
            if hasattr(self.phase2, 'blip_model') and self.phase2.blip_model is not None:
                if hasattr(self.phase2.blip_model, 'model_loaded') and self.phase2.blip_model.model_loaded:
                    self.phase2_available = True
                    logger.info("✅ Phase 2 (BLIP-2) loaded successfully - HEAVY MODEL ACTIVE")
                else:
                    # Force load BLIP model if not loaded
                    logger.info("🔄 Force loading BLIP-2 model...")
                    self.phase2.blip_model._lazy_load_blip()
                    if self.phase2.blip_model.model_loaded:
                        self.phase2_available = True
                        logger.info("✅ BLIP-2 force loaded successfully - HEAVY MODEL ACTIVE")
                    else:
                        raise RuntimeError("BLIP-2 model failed to load - HEAVY MODEL IMPLEMENTATION REQUIRES ALL MODELS")
            else:
                raise RuntimeError("Phase 2 BLIP model not initialized - HEAVY MODEL IMPLEMENTATION REQUIRES ALL MODELS")
            
            # Force load clip extractor
            logger.info("Loading Clip Extractor - HEAVY MODE")
            self.clip_extractor = ClipExtractor()
            logger.info("✅ Clip Extractor loaded successfully")
            
            self._models_loaded = True
            
            # Log final status
            logger.info("🎉 HEAVY MODEL IMPLEMENTATION COMPLETE - ALL MODELS LOADED:")
            logger.info("   ✅ OpenCLIP (Phase 1) - ACTIVE")
            logger.info("   ✅ BLIP-2 (Phase 2) - ACTIVE")
            logger.info("   ✅ Clip Extractor - ACTIVE")
            logger.info("   🚀 MAXIMUM ACCURACY MODE ENABLED")
            
            memory_manager.log_memory_usage("After heavy model loading")
            
        except Exception as e:
            logger.error(f"❌ HEAVY MODEL LOADING FAILED: {e}")
            logger.error("HEAVY MODEL IMPLEMENTATION REQUIRES ALL MODELS - NO FALLBACKS ALLOWED")
            raise RuntimeError(f"Heavy model implementation failed: {e}. All models must load successfully.")
    
    def _load_models(self):
        """Load all models with enhanced memory management and system optimization."""
        if self._models_loaded:
            return
            
        logger.info("Loading models with enhanced memory management and system optimization...")
        
        # Log initial system state
        system_info = system_optimizer.get_system_info()
        logger.info(f"Initial system state: {system_info}")
        
        # Log initial memory usage
        memory_manager.log_memory_usage("Before model loading")
        
        try:
            if self.heavy_models:
                logger.info("Heavy models enabled - using optimized progressive loading")
                
                # Use system optimizer context for heavy model operations
                with system_optimizer.optimized_context(enable_monitoring=True) as optimizations:
                    logger.info(f"Applied system optimizations: {optimizations}")
                    self._load_models_progressively()
            else:
                logger.info("Standard model loading with basic optimization")
                
                # Apply basic optimizations for standard loading
                basic_opts = system_optimizer.optimize_for_heavy_models()
                logger.info(f"Applied basic optimizations: {basic_opts}")
                
                try:
                    self._load_models_standard()
                finally:
                    system_optimizer.restore_original_settings()
                
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            logger.info("Attempting emergency fallback with minimal optimization...")
            
            # Ensure optimizations are restored before emergency loading
            system_optimizer.restore_original_settings()
            
            # Try emergency fallback
            try:
                logger.info("Attempting emergency model loading with memory optimization...")
                memory_manager.aggressive_cleanup()
                self._load_models_emergency()
                
            except Exception as emergency_e:
                logger.error(f"Emergency model loading also failed: {emergency_e}")
                raise RuntimeError(f"Failed to load models: {e}. Emergency loading also failed: {emergency_e}")
        
        # Log final system state and memory usage
        final_system_info = system_optimizer.get_system_info()
        logger.info(f"Final system state: {final_system_info}")
        memory_manager.log_memory_usage("After model loading")
    
    def _load_models_progressively(self):
        """Load models using progressive loading system."""
        # Register models for progressive loading
        progressive_loader.register_model(
            'phase1_mvp',
            lambda: Phase1MVP(),
            priority=1,  # Highest priority
            callback=self._on_phase1_loaded
        )
        
        progressive_loader.register_model(
            'phase2_reranker',
            lambda: Phase2Reranker(lazy_load=False),
            priority=2,
            dependencies=['phase1_mvp'],
            callback=self._on_phase2_loaded
        )
        
        progressive_loader.register_model(
            'clip_extractor',
            lambda: ClipExtractor(),
            priority=3,
            callback=self._on_clip_extractor_loaded
        )
        
        # Start progressive loading
        progressive_loader.start_progressive_loading()
        
        # Check loading results
        loading_status = progressive_loader.get_loading_status()
        
        # Set availability based on what was loaded
        self.phase1 = progressive_loader.get_model('phase1_mvp')
        self.phase2 = progressive_loader.get_model('phase2_reranker')
        self.phase2_available = self.phase2 is not None
        self.clip_extractor = progressive_loader.get_model('clip_extractor')
        
        if self.phase1 is None:
            raise RuntimeError("Failed to load critical Phase 1 model")
        
        if self.clip_extractor is None:
            raise RuntimeError("Failed to load clip extractor")
        
        self._models_loaded = True
        
        # Log final status
        loaded_count = sum(1 for status in loading_status.values() if status == 'loaded')
        total_count = len(loading_status)
        logger.info(f"Progressive loading completed: {loaded_count}/{total_count} models loaded")
        memory_manager.log_memory_usage("After progressive model loading")
    
    def _load_models_standard(self):
        """Load models using standard method (for lightweight implementation)."""
        logger.info("Loading Phase 1 (OpenCLIP)...")
        self.phase1 = Phase1MVP()
        
        logger.info("Loading Phase 2 (BLIP re-ranker)...")
        try:
            self.phase2 = Phase2Reranker(lazy_load=True)
            self.phase2_available = True
            logger.info("Phase 2 (BLIP re-ranker) initialized successfully")
        except Exception as e:
            logger.warning(f"Phase 2 initialization failed: {e}")
            logger.info("Running in MVP-only mode")
            self.phase2 = None
            self.phase2_available = False
        
        logger.info("Loading clip extractor...")
        self.clip_extractor = ClipExtractor()
        
        self._models_loaded = True
        logger.info("Standard model loading completed")
    
    def _load_models_emergency(self):
        """Emergency model loading with minimal configuration."""
        logger.info("Emergency loading: minimal configuration")
        
        # Try to load only essential components
        self.phase1 = Phase1MVP()
        self.phase2 = None
        self.phase2_available = False
        self.clip_extractor = ClipExtractor()
        
        self._models_loaded = True
        logger.info("Emergency model loading successful (MVP-only mode)")
    
    def _on_phase1_loaded(self, model):
        """Callback for Phase 1 model loading."""
        logger.info("Phase 1 (OpenCLIP) loaded successfully via progressive loading")
    
    def _on_phase2_loaded(self, model):
        """Callback for Phase 2 model loading."""
        logger.info("Phase 2 (BLIP re-ranker) loaded successfully via progressive loading")
    
    def _on_clip_extractor_loaded(self, model):
        """Callback for clip extractor loading."""
        logger.info("Clip extractor loaded successfully via progressive loading")
    
    def _ensure_models_loaded(self):
        """Ensure models are loaded before processing."""
        if not self._models_loaded:
            self._load_models()
    
    def preprocess_query(self, query: str) -> str:
        """Preprocess query to improve detection accuracy."""
        # Remove extra whitespace and normalize
        query = re.sub(r'\s+', ' ', query.strip())
        
        # Convert to lowercase for consistency
        query = query.lower()
        
        # Handle common query patterns and improvements
        query_improvements = {
            # Common action variations
            r'\bwalks?\b': 'walking',
            r'\bruns?\b': 'running', 
            r'\bjumps?\b': 'jumping',
            r'\bfalls?\b': 'falling',
            r'\bsits?\b': 'sitting',
            r'\bstands?\b': 'standing',
            r'\bdrives?\b': 'driving',
            r'\bhits?\b': 'hitting',
            r'\bcrashes?\b': 'crashing',
            
            # Common object variations
            r'\bautomobile\b': 'car',
            r'\bvehicle\b': 'car',
            r'\bpedestrian\b': 'person',
            r'\bindividual\b': 'person',
            r'\bcanine\b': 'dog',
            
            # Color standardization
            r'\bdark blue\b': 'navy',
            r'\blight blue\b': 'blue',
            r'\bdark green\b': 'green',
            r'\blight green\b': 'green',
        }
        
        # Apply improvements
        for pattern, replacement in query_improvements.items():
            query = re.sub(pattern, replacement, query)
        
        # Remove unnecessary articles and prepositions for better matching
        query = re.sub(r'\b(a|an|the)\s+', '', query)
        
        # Simplify complex sentences - keep main action and objects
        # Remove filler words that don't help with visual detection
        filler_words = ['very', 'really', 'quite', 'somewhat', 'rather', 'pretty']
        for word in filler_words:
            query = re.sub(rf'\b{word}\s+', '', query)
        
        logger.info(f"Query preprocessed: '{query}'")
        return query
    
    def process_query(self, video_path: str, query: str, mode: str = "mvp", 
                     top_k: Optional[int] = None, threshold: Optional[float] = None, debug_mode: bool = False) -> Dict:
        """Process a video query with specified mode and comprehensive error handling."""
        
        # Ensure models are loaded before processing
        try:
            self._ensure_models_loaded()
        except Exception as e:
            return {
                'status': 'error',
                'error': f"Failed to load required models: {str(e)}",
                'query': query,
                'mode': mode,
                'results': [],
                'error_type': 'model_loading_error'
            }
        
        if top_k is None:
            top_k = settings.TOP_K_RESULTS
        if threshold is None:
            threshold = settings.CONFIDENCE_THRESHOLD
        
        # Preprocess query for better detection
        original_query = query
        processed_query = self.preprocess_query(query)
        
        logger.info(f"Processing query: '{original_query}' -> '{processed_query}' on {video_path} with mode: {mode}, debug: {debug_mode}")
        
        try:
            # Validate video file first
            validation_result = self.validate_video(video_path)
            if not validation_result['valid']:
                return {
                    'status': 'error',
                    'error': f"Video validation failed: {validation_result['error']}",
                    'query': original_query,
                    'mode': mode,
                    'results': []
                }
            # Select processing pipeline
            if mode == "mvp":
                result = self.phase1.process_video(video_path, processed_query, top_k, debug_mode=debug_mode)
                # Handle debug mode return format
                if debug_mode and isinstance(result, tuple):
                    results, debug_info = result
                    logger.info(f"Debug info collected for {len(debug_info)} windows")
                else:
                    results = result
            elif mode == "reranked":
                if self.phase2_available:
                    results = self.phase2.process_video(video_path, processed_query, top_k, debug_mode=debug_mode)
                else:
                    logger.warning("Phase 2 not available, falling back to MVP mode")
                    result = self.phase1.process_video(video_path, processed_query, top_k, debug_mode=debug_mode)
                    # Handle debug mode return format
                    if debug_mode and isinstance(result, tuple):
                        results, debug_info = result
                        logger.info(f"Debug info collected for {len(debug_info)} windows")
                    else:
                        results = result
            elif mode == "advanced":
                if self.phase2_available:
                    # For now, use phase2 as advanced mode (Phase 3 can be added later)
                    results = self.phase2.process_video(video_path, processed_query, top_k, debug_mode=debug_mode)
                else:
                    logger.warning("Advanced mode not available, falling back to MVP mode")
                    result = self.phase1.process_video(video_path, processed_query, top_k, debug_mode=debug_mode)
                    # Handle debug mode return format
                    if debug_mode and isinstance(result, tuple):
                        results, debug_info = result
                        logger.info(f"Debug info collected for {len(debug_info)} windows")
                    else:
                        results = result
            else:
                raise ValueError(f"Unknown processing mode: {mode}")
            
            # Filter by threshold with proper type checking
            filtered_results = []
            for result in results:
                # Ensure result is a dictionary and has required keys
                if isinstance(result, dict) and 'confidence' in result and 'timestamp' in result:
                    if result['confidence'] >= threshold:
                        filtered_results.append(result)
                else:
                    logger.warning(f"Invalid result format: {type(result)} - {result}")
            
            # Extract clips for results
            for result in filtered_results:
                try:
                    # Double-check result structure before accessing
                    if isinstance(result, dict) and 'timestamp' in result:
                        clip_path = self.clip_extractor.extract_clip_with_padding(
                            video_path, 
                            result['timestamp'],
                            settings.CLIP_DURATION
                        )
                        result['clip_path'] = clip_path
                    else:
                        logger.warning(f"Invalid result structure for clip extraction: {result}")
                        result['clip_path'] = None
                except Exception as e:
                    timestamp = result.get('timestamp', 'unknown') if isinstance(result, dict) else 'unknown'
                    logger.warning(f"Failed to extract clip for timestamp {timestamp}: {e}")
                    if isinstance(result, dict):
                        result['clip_path'] = None
            
            response = {
                'status': 'success',
                'query': original_query,
                'processed_query': processed_query,
                'mode': mode,
                'results': filtered_results,
                'total_found': len(filtered_results)
            }
            
            # Add debug info if available
            if debug_mode and 'debug_info' in locals():
                response['debug_info'] = debug_info
            
            return response
            
        except MemoryError as e:
            logger.error(f"Memory allocation error during processing: {e}")
            return {
                'status': 'error',
                'error': f"Insufficient memory to process video. Try using a smaller video or restart the application. Details: {str(e)}",
                'query': original_query,
                'mode': mode,
                'results': [],
                'error_type': 'memory_error'
            }
        except OSError as e:
            if "paging file" in str(e).lower() or "1455" in str(e):
                logger.error(f"Windows paging file error: {e}")
                return {
                    'status': 'error',
                    'error': "System memory error (paging file too small). Please increase virtual memory or restart the application.",
                    'query': original_query,
                    'mode': mode,
                    'results': [],
                    'error_type': 'paging_file_error'
                }
            else:
                logger.error(f"System error during processing: {e}")
                return {
                    'status': 'error',
                    'error': f"System error: {str(e)}",
                    'query': original_query,
                    'mode': mode,
                    'results': [],
                    'error_type': 'system_error'
                }
        except FileNotFoundError as e:
            logger.error(f"File not found error: {e}")
            return {
                'status': 'error',
                'error': f"Required file not found: {str(e)}. Please check FFmpeg installation and file paths.",
                'query': original_query,
                'mode': mode,
                'results': [],
                'error_type': 'file_not_found'
            }
        except Exception as e:
            logger.error(f"Unexpected error processing query: {e}")
            return {
                'status': 'error',
                'error': f"Unexpected error: {str(e)}",
                'query': original_query,
                'mode': mode,
                'results': [],
                'error_type': 'unknown_error'
            }
    
    def validate_video(self, video_path: str) -> Dict:
        """Validate video file format and accessibility."""
        try:
            video_file = Path(video_path)
            
            if not video_file.exists():
                return {'valid': False, 'error': 'Video file does not exist'}
            
            file_extension = video_file.suffix.lower().lstrip('.')
            if file_extension not in settings.SUPPORTED_FORMATS:
                return {
                    'valid': False, 
                    'error': f'Unsupported format: {file_extension}. Supported: {settings.SUPPORTED_FORMATS}'
                }
            
            file_size = video_file.stat().st_size
            if file_size > settings.MAX_VIDEO_SIZE:
                return {
                    'valid': False, 
                    'error': f'Video file too large: {file_size} bytes. Max: {settings.MAX_VIDEO_SIZE} bytes'
                }
            
            return {
                'valid': True, 
                'format': file_extension,
                'size': file_size,
                'path': str(video_file)
            }
            
        except Exception as e:
            return {'valid': False, 'error': f'Error validating video: {str(e)}'}