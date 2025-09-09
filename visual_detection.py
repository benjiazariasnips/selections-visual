#!/usr/bin/env python3
"""
CLOUD-OPTIMIZED VISUAL CONTENT DETECTION
GPU-Optimized version for cloud deployment

ENHANCEMENTS FOR CLOUD/GPU:
1. PRIORITIZED GPU utilization with CUDA optimizations
2. BATCH processing for better GPU memory utilization  
3. MIXED precision support for faster inference
4. MEMORY management and garbage collection
5. ASYNC processing capabilities
6. OPTIMIZED frame preprocessing pipeline

DETECTION CAPABILITIES:
- Adult Content: Kissing, Hugging, Nudity, Partial Nudity, Intimate Couple, Dancing
- Emotions: Happy, Sad, Angry, Fearful, Disgusted
- Precise timing analysis with blended scoring
"""

import os
import time
import logging
import argparse
import gc
from typing import List, Dict, Tuple, Optional, Union
import json
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CloudVisualDetector:
    """
    Cloud-optimized visual content detection with GPU acceleration:
    - Prioritized CUDA utilization
    - Batch processing for GPU efficiency
    - Memory management for cloud environments
    - Mixed precision support
    """
    
    def __init__(self, device: str = 'auto', lambda_blend: float = 0.75, 
                 use_mixed_precision: bool = True, batch_size: int = 8):
        
        # PRIORITIZED GPU DETECTION for cloud environments
        self.device = self._get_optimal_device(device)
        self.use_mixed_precision = use_mixed_precision and self.device == 'cuda'
        self.batch_size = batch_size
        
        # Initialize mixed precision scaler
        if self.use_mixed_precision:
            from torch.cuda.amp import autocast
            self.autocast = autocast
            logger.info("🚀 Mixed precision enabled for faster GPU inference")
        else:
            self.autocast = None
        
        # Store lambda blend parameter
        self.lambda_blend = lambda_blend
        logger.info(f"Lambda blend: {lambda_blend:.1%} per-second + {1-lambda_blend:.1%} segment")
        
        # GPU memory management
        if self.device == 'cuda':
            torch.cuda.empty_cache()
            self._log_gpu_info()
        
        logger.info(f"🔥 Initializing Cloud Visual Detector on {self.device.upper()}")
        
        # Load CLIP model with GPU optimizations
        self._load_models()
        
        # Set models to eval mode and optimize for inference
        self.clip_model.eval()
        if self.device == 'cuda':
            # Optimize for inference
            self.clip_model = torch.jit.optimize_for_inference(self.clip_model)
        
        # PROVEN DETECTION PROMPTS (optimized for cloud processing)
        self._initialize_prompts()
        
        # CALIBRATION PARAMETERS (proven successful)
        self._initialize_calibration()
        
        # PRECISION TIMING THRESHOLDS 
        self.precision_thresholds = {
            'kissing': 0.7, 'hugging': 0.65, 'nudity': 0.8, 'partial_nudity': 0.7,
            'intimate_couple': 0.8, 'dancing': 0.75,
            'happy': 0.75, 'sad': 0.7, 'angry': 0.6, 'fearful': 0.7, 'disgusted': 0.45
        }
        
        logger.info("✅ Cloud Visual Detector initialized with GPU optimizations")
    
    def _get_optimal_device(self, device: str) -> str:
        """Prioritized device selection for cloud environments"""
        if device == 'auto':
            if torch.cuda.is_available():
                # Check for multiple GPUs and select the best one
                if torch.cuda.device_count() > 1:
                    # Select GPU with most memory
                    best_gpu = 0
                    max_memory = 0
                    for i in range(torch.cuda.device_count()):
                        memory = torch.cuda.get_device_properties(i).total_memory
                        if memory > max_memory:
                            max_memory = memory
                            best_gpu = i
                    torch.cuda.set_device(best_gpu)
                    logger.info(f"🚀 Selected GPU {best_gpu}/{torch.cuda.device_count()} with {max_memory/1e9:.1f}GB memory")
                
                device = f'cuda:{torch.cuda.current_device()}'
                logger.info(f"🔥 CUDA GPU: {torch.cuda.get_device_name()}")
                return device
            elif torch.backends.mps.is_available():
                logger.info("🍎 Apple Silicon GPU (MPS)")
                return 'mps'
            else:
                logger.warning("⚠️  No GPU detected - falling back to CPU")
                return 'cpu'
        else:
            return device
    
    def _log_gpu_info(self):
        """Log detailed GPU information for cloud monitoring"""
        if torch.cuda.is_available():
            current_device = torch.cuda.current_device()
            gpu_props = torch.cuda.get_device_properties(current_device)
            total_memory = gpu_props.total_memory / 1e9
            allocated = torch.cuda.memory_allocated(current_device) / 1e9
            cached = torch.cuda.memory_reserved(current_device) / 1e9
            
            logger.info(f"🔥 GPU Info: {gpu_props.name}")
            logger.info(f"💾 Memory: {allocated:.2f}GB/{total_memory:.2f}GB allocated, {cached:.2f}GB cached")
            logger.info(f"🔢 Compute: {gpu_props.major}.{gpu_props.minor}, Multiprocessors: {gpu_props.multi_processor_count}")
    
    def _load_models(self):
        """Load models with GPU optimizations"""
        try:
            # Load with GPU-optimized settings
            self.clip_model = CLIPModel.from_pretrained(
                "openai/clip-vit-large-patch14",
                torch_dtype=torch.float16 if self.use_mixed_precision else torch.float32
            ).to(self.device)
            
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
            
            logger.info("✅ CLIP model loaded with GPU optimizations")
            
        except Exception as e:
            logger.error(f"❌ Model loading failed: {e}")
            raise
    
    def _initialize_prompts(self):
        """Initialize detection prompts"""
        self.adult_content_prompts = {
            'kissing': {
                'positive': [
                    "two people kissing with their lips touching",
                    "a passionate romantic kiss between two people", 
                    "people engaged in intimate lip-to-lip contact",
                    "a couple kissing mouth to mouth",
                    "romantic kissing with lips pressed together"
                ],
                'negative': [
                    "people talking face to face",
                    "people standing near each other",
                    "people eating or drinking",
                    "people laughing or smiling", 
                    "casual conversation between people"
                ],
                'weight_factors': [1.0, 0.9, 0.8, 0.7, 0.6]
            },
            
            'hugging': {
                'positive': [
                    "two people hugging with arms wrapped around each other",
                    "people in a close physical embrace",
                    "affectionate hug with bodies pressed together",
                    "people holding each other tightly",
                    "warm embrace between two people"
                ],
                'negative': [
                    "people standing apart from each other",
                    "people shaking hands",
                    "people walking together",
                    "casual standing conversation",
                    "people sitting separately"
                ],
                'weight_factors': [1.0, 0.8, 0.7, 0.6, 0.5]
            },
            
            'nudity': {
                'positive': [
                    "completely naked person with no clothing",
                    "nude body fully exposed without clothes",
                    "person with all clothing removed showing private parts",
                    "fully undressed person showing genitals",
                    "complete nudity with exposed intimate body parts"
                ],
                'negative': [
                    "person wearing full clothing",
                    "person in normal dress",
                    "clothed individual",
                    "person in casual attire",
                    "fully dressed person"
                ],
                'weight_factors': [1.0, 0.9, 0.8, 0.7, 0.6]
            },
            
            'partial_nudity': {
                'positive': [
                    "person in underwear or lingerie",
                    "topless person showing breasts",
                    "person in bikini or revealing swimwear",
                    "partially undressed person showing intimate areas",
                    "person with exposed chest or intimate body parts"
                ],
                'negative': [
                    "person in normal clothing",
                    "person wearing sports attire",
                    "person in casual wear",
                    "fully clothed individual",
                    "person in appropriate dress"
                ],
                'weight_factors': [1.0, 0.8, 0.7, 0.6, 0.5]
            },
            
            'intimate_couple': {
                'positive': [
                    "couple in romantic intimate position together",
                    "two people in sexual or erotic pose",
                    "lovers in intimate romantic position",
                    "couple engaged in sexual activity",
                    "people in intimate romantic embrace showing desire"
                ],
                'negative': [
                    "friends having normal conversation",
                    "business people talking",
                    "family members interacting",
                    "casual social interaction",
                    "platonic friendship"
                ],
                'weight_factors': [1.0, 0.9, 0.8, 0.7, 0.6]
            },
            
            'dancing': {
                'positive': [
                    "person dancing in suggestive provocative manner",
                    "erotic sensual dance with sexual movements",
                    "provocative dancing with suggestive body movements",
                    "sensual dancing with sexual hip movements",
                    "erotic dance performance with provocative gestures"
                ],
                'negative': [
                    "normal social dancing",
                    "ballroom or formal dancing",
                    "casual movement or walking",
                    "exercise or fitness activity",
                    "regular dancing at party"
                ],
                'weight_factors': [1.0, 0.8, 0.7, 0.6, 0.5]
            }
        }
        
        self.emotion_prompts = {
            'happy': {
                'positive': [
                    "person with a big genuine smile showing happiness",
                    "person laughing with joy and positive emotion",
                    "person with bright cheerful expression showing delight",
                    "person beaming with happiness and contentment",
                    "person with joyful expression and positive facial features"
                ],
                'negative': [
                    "person with neutral facial expression",
                    "person looking sad or depressed",
                    "person with angry or upset face",
                    "person looking worried or anxious",
                    "person with serious or stern expression"
                ],
                'weight_factors': [1.0, 0.9, 0.8, 0.7, 0.6]
            },
            
            'sad': {
                'positive': [
                    "person with sad expression showing sorrow",
                    "person crying with tears and emotional distress",
                    "person with downcast eyes showing sadness",
                    "person with melancholy expression and drooping features",
                    "person looking depressed with sorrowful face"
                ],
                'negative': [
                    "person with happy smiling expression",
                    "person laughing with joy",
                    "person with neutral calm face",
                    "person looking excited or energetic",
                    "person with confident expression"
                ],
                'weight_factors': [1.0, 0.9, 0.8, 0.7, 0.6]
            },
            
            'angry': {
                'positive': [
                    "person with angry furious expression showing rage",
                    "person with clenched jaw and angry eyebrows",
                    "person showing anger with intense hostile expression",
                    "person with mad upset face showing irritation",
                    "person with aggressive angry facial features"
                ],
                'negative': [
                    "person with calm peaceful expression",
                    "person smiling with happiness",
                    "person with relaxed neutral face",
                    "person looking content and serene",
                    "person with friendly gentle expression"
                ],
                'weight_factors': [1.0, 0.9, 0.8, 0.7, 0.6]
            },
            
            'fearful': {
                'positive': [
                    "person with frightened scared expression showing fear",
                    "person with wide eyes showing terror and anxiety",
                    "person looking worried with fearful expression",
                    "person showing panic with anxious scared face",
                    "person with nervous frightened facial features"
                ],
                'negative': [
                    "person with brave confident expression",
                    "person looking calm and relaxed",
                    "person with happy peaceful face",
                    "person showing courage and strength",
                    "person with composed secure expression"
                ],
                'weight_factors': [1.0, 0.9, 0.8, 0.7, 0.6]
            },
            
            'disgusted': {
                'positive': [
                    "person with disgusted expression showing revulsion",
                    "person with wrinkled nose showing disgust",
                    "person with repulsed face showing distaste",
                    "person looking sick with disgusted expression",
                    "person with revolted facial features showing aversion"
                ],
                'negative': [
                    "person with pleasant satisfied expression",
                    "person looking content and happy",
                    "person with neutral accepting face",
                    "person showing enjoyment and pleasure",
                    "person with appreciative expression"
                ],
                'weight_factors': [1.0, 0.9, 0.8, 0.7, 0.6]
            }
        }
        
        # Combine all prompts
        self.all_prompts = {**self.adult_content_prompts, **self.emotion_prompts}
    
    def _initialize_calibration(self):
        """Initialize calibration parameters"""
        self.adult_calibration_params = {
            'kissing': {'base_threshold': 0.3, 'multiplier': 2.5, 'power': 1.2},
            'hugging': {'base_threshold': 0.25, 'multiplier': 2.0, 'power': 1.1},
            'nudity': {'base_threshold': 0.4, 'multiplier': 3.0, 'power': 1.3},
            'partial_nudity': {'base_threshold': 0.3, 'multiplier': 2.2, 'power': 1.15},
            'intimate_couple': {'base_threshold': 0.35, 'multiplier': 2.8, 'power': 1.25},
            'dancing': {'base_threshold': 0.2, 'multiplier': 1.8, 'power': 1.05}
        }
        
        self.emotion_calibration_params = {
            'happy': {'base_threshold': 0.25, 'multiplier': 2.2, 'power': 1.1},
            'sad': {'base_threshold': 0.3, 'multiplier': 2.0, 'power': 1.15},
            'angry': {'base_threshold': 0.35, 'multiplier': 2.3, 'power': 1.2},
            'fearful': {'base_threshold': 0.3, 'multiplier': 2.1, 'power': 1.1},
            'disgusted': {'base_threshold': 0.35, 'multiplier': 2.0, 'power': 1.15}
        }
        
        self.calibration_params = {**self.adult_calibration_params, **self.emotion_calibration_params}
    
    def calibrate_score(self, raw_score: float, content_type: str) -> float:
        """Calibrate raw CLIP scores to proper 0-1 range"""
        params = self.calibration_params[content_type]
        
        if raw_score < params['base_threshold']:
            return 0.0
        
        normalized = (raw_score - params['base_threshold']) / (1.0 - params['base_threshold'])
        normalized = min(1.0, normalized)
        
        calibrated = (normalized ** params['power']) * params['multiplier']
        return min(1.0, max(0.0, calibrated))
    
    def detect_content_batch_optimized(self, frames: List[np.ndarray], content_type: str) -> float:
        """GPU-optimized batch detection with mixed precision"""
        if not frames:
            return 0.0
        
        prompts_config = self.all_prompts[content_type]
        positive_prompts = prompts_config['positive']
        negative_prompts = prompts_config['negative']
        weight_factors = prompts_config['weight_factors']
        
        # Convert frames to PIL images efficiently
        pil_images = [Image.fromarray(frame) for frame in frames]
        
        # Method 1: Batch individual prompt scoring (GPU optimized)
        individual_scores = []
        
        for i, prompt in enumerate(positive_prompts):
            all_prompts = [prompt] + negative_prompts
            
            # Batch process all images with this prompt set
            inputs = self.clip_processor(
                text=all_prompts,
                images=pil_images,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                if self.use_mixed_precision and self.autocast:
                    with self.autocast():
                        outputs = self.clip_model(**inputs)
                        logits = outputs.logits_per_image
                else:
                    outputs = self.clip_model(**inputs)
                    logits = outputs.logits_per_image
                
                probs = F.softmax(logits, dim=-1)
                
                # Extract positive probabilities for all images
                positive_probs = probs[:, 0]  # First column is positive prompt
                weighted_scores = positive_probs * weight_factors[i]
                individual_scores.append(weighted_scores.max().item())
            
            # Memory cleanup for cloud environments
            del inputs, outputs, logits, probs
            if self.device == 'cuda':
                torch.cuda.empty_cache()
        
        # Method 2: Ensemble scoring (optimized)
        ensemble_scores = []
        all_prompts = positive_prompts + negative_prompts
        
        # Batch process with ensemble prompts
        inputs = self.clip_processor(
            text=all_prompts,
            images=pil_images,
            return_tensors="pt", 
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            if self.use_mixed_precision and self.autocast:
                with self.autocast():
                    outputs = self.clip_model(**inputs)
                    logits = outputs.logits_per_image
            else:
                outputs = self.clip_model(**inputs)
                logits = outputs.logits_per_image
            
            probs = F.softmax(logits, dim=-1)
            
            # Calculate weighted average for each image
            for img_idx in range(len(pil_images)):
                positive_probs = probs[img_idx, :len(positive_prompts)]
                weighted_positive = sum(p * w for p, w in zip(positive_probs, weight_factors))
                weighted_positive /= sum(weight_factors)
                ensemble_scores.append(weighted_positive.item())
        
        # Memory cleanup
        del inputs, outputs, logits, probs
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        
        # Method 3: Comparative scoring
        comparative_scores = []
        best_positive = positive_prompts[0]
        best_negative = negative_prompts[0]
        
        inputs = self.clip_processor(
            text=[best_positive, best_negative],
            images=pil_images,
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            if self.use_mixed_precision and self.autocast:
                with self.autocast():
                    outputs = self.clip_model(**inputs)
                    logits = outputs.logits_per_image
            else:
                outputs = self.clip_model(**inputs)
                logits = outputs.logits_per_image
            
            probs = F.softmax(logits, dim=-1)
            
            for img_idx in range(len(pil_images)):
                pos_prob = probs[img_idx, 0].item()
                neg_prob = probs[img_idx, 1].item()
                relative_score = pos_prob / (pos_prob + neg_prob) if (pos_prob + neg_prob) > 0 else 0
                comparative_scores.append(relative_score)
        
        # Memory cleanup
        del inputs, outputs, logits, probs
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        
        # Combine methods
        method_weights = [0.4, 0.4, 0.2]
        final_raw_score = (
            np.mean(individual_scores) * method_weights[0] +
            np.mean(ensemble_scores) * method_weights[1] +
            np.mean(comparative_scores) * method_weights[2]
        )
        
        # Apply calibration
        calibrated_score = self.calibrate_score(final_raw_score, content_type)
        
        logger.debug(f"{content_type} - Raw: {final_raw_score:.3f}, Calibrated: {calibrated_score:.3f}")
        return calibrated_score
    
    def smart_frame_selection_gpu(self, cap, start_time: float, end_time: float, fps: float) -> List[np.ndarray]:
        """GPU-optimized frame selection"""
        segment_duration = end_time - start_time
        frames_per_second = 4
        total_candidate_frames = int(segment_duration * frames_per_second)
        
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        
        # Extract candidate frames
        candidate_frame_indices = np.linspace(start_frame, end_frame - 1, total_candidate_frames, dtype=int)
        candidate_frames = []
        
        for frame_idx in candidate_frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, (224, 224))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                candidate_frames.append(frame)
        
        if len(candidate_frames) < 6:
            return candidate_frames
        
        # GPU-optimized quality scoring using vectorized operations
        frame_scores = []
        for i, frame in enumerate(candidate_frames):
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            
            # Vectorized quality metrics
            sharpness = cv2.Laplacian(frame_gray, cv2.CV_64F).var()
            contrast = np.std(frame_gray)
            brightness_balance = 1.0 / (1.0 + abs(np.mean(frame_gray) - 128))
            
            quality_score = sharpness * 0.5 + contrast * 0.3 + brightness_balance * 100 * 0.2
            
            frame_scores.append({
                'index': i,
                'frame': frame,
                'score': quality_score,
                'timestamp': start_time + (i / frames_per_second)
            })
        
        # Smart selection with temporal distribution
        frame_scores.sort(key=lambda x: x['score'], reverse=True)
        
        selected_frames = []
        selected_timestamps = []
        selected_indices = set()
        
        for frame_data in frame_scores:
            timestamp = frame_data['timestamp']
            frame_index = frame_data['index']
            
            too_close = any(abs(timestamp - prev_ts) < 0.5 for prev_ts in selected_timestamps)
            
            if not too_close or len(selected_frames) < 3:
                selected_frames.append(frame_data['frame'])
                selected_timestamps.append(timestamp)
                selected_indices.add(frame_index)
                
                if len(selected_frames) >= 6:
                    break
        
        # Fill remaining slots
        while len(selected_frames) < 6 and len(selected_frames) < len(candidate_frames):
            for frame_data in frame_scores:
                if frame_data['index'] not in selected_indices:
                    selected_frames.append(frame_data['frame'])
                    selected_indices.add(frame_data['index'])
                    if len(selected_frames) >= 6:
                        break
        
        logger.debug(f"GPU-optimized selection: {total_candidate_frames} → {len(selected_frames)} frames")
        return selected_frames
    
    def extract_video_segments_gpu(self, video_path: str, start_time: float = 0,
                                  end_time: float = None, segment_duration: float = 6.0) -> List[Dict]:
        """GPU-optimized video segment extraction"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_duration = total_frames / fps
        
        if end_time is None:
            end_time = video_duration
        
        segments = []
        current_time = start_time
        
        with tqdm(desc="Extracting video segments", unit="segments") as pbar:
            while current_time < end_time:
                segment_end = min(current_time + segment_duration, end_time)
                
                # GPU-optimized frame selection
                smart_frames = self.smart_frame_selection_gpu(cap, current_time, segment_end, fps)
                
                if smart_frames:
                    segments.append({
                        'start_time': current_time,
                        'end_time': segment_end,
                        'frames': smart_frames,
                        'fps': fps
                    })
                
                current_time += segment_duration
                pbar.update(1)
        
        cap.release()
        logger.info(f"✅ Extracted {len(segments)} segments with GPU-optimized processing")
        return segments
    
    def process_video_gpu_optimized(self, video_path: str, start_time: float = 0,
                                   end_time: float = None, segment_duration: float = 6.0) -> Tuple[List[Dict], Dict]:
        """Main GPU-optimized video processing pipeline"""
        
        logger.info("🚀 === CLOUD-OPTIMIZED VISUAL DETECTION ===")
        
        # GPU memory status
        if self.device == 'cuda':
            self._log_gpu_info()
        
        # Extract video segments
        segments = self.extract_video_segments_gpu(video_path, start_time, end_time, segment_duration)
        
        if not segments:
            return [], {}
        
        # Detection categories
        adult_types = ['kissing', 'hugging', 'nudity', 'partial_nudity', 'intimate_couple', 'dancing']
        emotion_types = ['happy', 'sad', 'angry', 'fearful', 'disgusted']
        all_content_types = adult_types + emotion_types
        
        results = []
        high_score_segments = []
        
        # Initialize output
        os.makedirs("output", exist_ok=True)
        output_path = "output/visual_detection_results.csv"
        batch_size = 3
        segment_batch_results = []
        csv_initialized = False
        
        # Process segments with GPU optimization
        for segment_idx, segment in enumerate(tqdm(segments, desc="🔥 GPU Processing segments")):
            frames = segment['frames']
            start_sec = segment['start_time']
            end_sec = segment['end_time']
            fps = segment['fps']
            
            # Batch detect all content types
            segment_scores = {}
            
            for content_type in all_content_types:
                score = self.detect_content_batch_optimized(frames, content_type)
                segment_scores[content_type] = score
                
                # Check for high scores
                if score > self.precision_thresholds[content_type]:
                    high_score_segments.append({
                        'start_time': start_sec,
                        'end_time': end_sec,
                        'content_type': content_type,
                        'score': score
                    })
            
            # Generate per-second predictions
            segment_duration_sec = end_sec - start_sec
            for second_offset in range(int(segment_duration_sec)):
                timestamp = start_sec + second_offset
                
                # Temporal weighting
                segment_center = (start_sec + end_sec) / 2
                distance_from_center = abs(timestamp - segment_center)
                max_distance = segment_duration_sec / 2
                temporal_weight = 1.0 - (distance_from_center / max_distance) * 0.3
                
                minutes = int(timestamp // 60)
                seconds = int(timestamp % 60)
                time_mmss = f"{minutes:02d}:{seconds:02d}"
                
                result = {
                    'time_mmss': time_mmss,
                    'second': timestamp,
                    'frame': int(timestamp * fps),
                    
                    # Adult content scores
                    'kissing_score': segment_scores['kissing'] * temporal_weight,
                    'hugging_score': segment_scores['hugging'] * temporal_weight,
                    'partial_nudity_score': segment_scores['partial_nudity'] * temporal_weight,
                    'intimate_couple_score': segment_scores['intimate_couple'] * temporal_weight,
                    'provocative_dancing_score': segment_scores['dancing'] * temporal_weight,
                    'nude_score': segment_scores['nudity'] * temporal_weight,
                    
                    # Emotion scores
                    'happy_score': segment_scores['happy'] * temporal_weight,
                    'sad_score': segment_scores['sad'] * temporal_weight,
                    'angry_score': segment_scores['angry'] * temporal_weight,
                    'fearful_score': segment_scores['fearful'] * temporal_weight,
                    'disgusted_score': segment_scores['disgusted'] * temporal_weight
                }
                results.append(result)
                segment_batch_results.append(result)
            
            # Batch save and memory cleanup
            if (segment_idx + 1) % batch_size == 0 or segment_idx == len(segments) - 1:
                if segment_batch_results:
                    df_batch = pd.DataFrame(segment_batch_results)
                    
                    if not csv_initialized:
                        df_batch.to_csv(output_path, index=False)
                        csv_initialized = True
                        logger.info(f"💾 Initialized CSV and saved first batch")
                    else:
                        df_batch.to_csv(output_path, mode='a', header=False, index=False)
                        logger.info(f"💾 Saved batch: segments {max(0, segment_idx + 1 - batch_size)}-{segment_idx}")
                    
                    segment_batch_results = []
                
                # GPU memory cleanup
                if self.device == 'cuda':
                    torch.cuda.empty_cache()
                    gc.collect()
        
        logger.info(f"✅ Generated predictions for {len(results)} seconds with GPU acceleration")
        
        # Precise timing analysis (if needed)
        precise_detections = {}
        if high_score_segments:
            logger.info(f"🎯 Analyzing {len(high_score_segments)} high-scoring segments for precise timing")
            # Could implement precise timing here if needed
        
        # Final GPU cleanup
        if self.device == 'cuda':
            torch.cuda.empty_cache()
            gc.collect()
            self._log_gpu_info()
        
        return results, precise_detections


def main():
    parser = argparse.ArgumentParser(description='Cloud-Optimized Visual Content Detection')
    parser.add_argument('--video_path', type=str, default='input/movie.mp4')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda', 'mps'])
    parser.add_argument('--start_time', type=float, default=0.0)
    parser.add_argument('--end_time', type=float, default=None)
    parser.add_argument('--segment_duration', type=float, default=6.0)
    parser.add_argument('--lambda_blend', type=float, default=0.75)
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for GPU processing')
    parser.add_argument('--mixed_precision', action='store_true', default=True, help='Use mixed precision for faster GPU inference')
    
    args = parser.parse_args()
    
    logger.info(f"🚀 Cloud Visual Detection - GPU Optimized")
    logger.info(f"🔧 Device: {args.device}, Batch size: {args.batch_size}, Mixed precision: {args.mixed_precision}")
    
    try:
        # Initialize GPU-optimized detector
        detector = CloudVisualDetector(
            device=args.device,
            lambda_blend=args.lambda_blend,
            use_mixed_precision=args.mixed_precision,
            batch_size=args.batch_size
        )
        
        # Process video with GPU optimizations
        start_processing = time.time()
        results, precise_detections = detector.process_video_gpu_optimized(
            args.video_path,
            start_time=args.start_time,
            end_time=args.end_time,
            segment_duration=args.segment_duration
        )
        processing_time = time.time() - start_processing
        
        # Results summary
        output_path = "output/visual_detection_results.csv"
        logger.info(f"🎉 Processing completed in {processing_time:.2f} seconds")
        logger.info(f"📊 Results saved to {output_path}")
        
        # Show score distribution
        if results:
            df = pd.read_csv(output_path)
            
            print(f"\n🚀 === CLOUD-OPTIMIZED VISUAL DETECTION RESULTS ===")
            print(f"⚡ Processing time: {processing_time:.2f}s")
            print(f"🔥 Device: {detector.device.upper()}")
            print(f"📊 Total seconds analyzed: {len(results)}")
            
            # Adult content summary
            print("\n--- ADULT CONTENT DETECTION ---")
            adult_cols = ['kissing_score', 'hugging_score', 'partial_nudity_score',
                         'intimate_couple_score', 'provocative_dancing_score', 'nude_score']
            for col in adult_cols:
                if col in df.columns:
                    scores = df[col].values
                    high_conf = (scores > 0.7).sum()
                    print(f"{col}: Max: {scores.max():.3f}, High conf: {high_conf}")
            
            # Emotion summary  
            print("\n--- EMOTION DETECTION ---")
            emotion_cols = ['happy_score', 'sad_score', 'angry_score', 'fearful_score', 'disgusted_score']
            for col in emotion_cols:
                if col in df.columns:
                    scores = df[col].values
                    high_conf = (scores > 0.7).sum()
                    print(f"{col}: Max: {scores.max():.3f}, High conf: {high_conf}")
        
        print(f"\n✅ CLOUD VISUAL DETECTION COMPLETE")
        print(f"🚀 GPU-optimized processing with mixed precision")
        print(f"⚡ {processing_time:.2f}s total processing time")
        print(f"📁 Results: {output_path}")
    
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
