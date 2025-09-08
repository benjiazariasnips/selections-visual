#!/usr/bin/env python3
"""
COMPREHENSIVE CONTENT & EMOTION DETECTION V2

IMPROVEMENTS OVER V1:
1. REMOVED "surprised" emotion detection (poor performance: only 2 high confidence detections)
2. ADDED precise timing detection for high-scoring segments
3. Two-phase approach: 6-second segments + precise 1-second analysis for peaks

Combines the successful adult detection approach from smart_selection_6s.py
with improved emotion detection and precise timing:

ADULT FEATURES (from successful smart_selection_6s.py):
- Kissing, Hugging, Nudity, Partial Nudity, Intimate Couple, Dancing

EMOTION DETECTION (improved):
- Happy, Sad, Angry, Fearful, Disgusted (removed Surprised)

PRECISION TIMING:
- When segment score > threshold, analyze individual seconds within that segment
- Provides exact timing for high-confidence detections
"""

import os
import time
import logging
import argparse
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
from sklearn.metrics import classification_report, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ComprehensiveContentDetectorV2:
    """
    Enhanced comprehensive video content and emotion detection with:
    - Proven adult content detection (from smart_selection_6s.py)
    - Improved emotion detection (removed surprised)
    - Precise timing detection for high-scoring segments
    """
    
    def __init__(self, device: str = 'auto', lambda_blend: float = 0.75):
        # Auto-detect best device
        if device == 'auto':
            if torch.cuda.is_available():
                self.device = 'cuda'
                logger.info(f"Auto-detected CUDA GPU: {torch.cuda.get_device_name()}")
            elif torch.backends.mps.is_available():
                self.device = 'mps'
                logger.info("Auto-detected Apple Silicon GPU (MPS)")
            else:
                self.device = 'cpu'
                logger.info("Auto-detected CPU")
        else:
            self.device = device
        
        # Store lambda blend parameter
        self.lambda_blend = lambda_blend
        logger.info(f"Lambda blend parameter: {lambda_blend} (per-second weight: {lambda_blend:.1%}, segment weight: {1-lambda_blend:.1%})")
        
        logger.info(f"Initializing Comprehensive Content & Emotion Detector V2 on {self.device}")
        
        # Load CLIP model
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        
        # PROVEN ADULT CONTENT PROMPTS (from successful smart_selection_6s.py)
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
        
        # IMPROVED EMOTION DETECTION PROMPTS (removed surprised)
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
        
        # PROVEN CALIBRATION PARAMETERS (from successful smart_selection_6s.py)
        self.adult_calibration_params = {
            'kissing': {'base_threshold': 0.3, 'multiplier': 2.5, 'power': 1.2},
            'hugging': {'base_threshold': 0.25, 'multiplier': 2.0, 'power': 1.1},
            'nudity': {'base_threshold': 0.4, 'multiplier': 3.0, 'power': 1.3},
            'partial_nudity': {'base_threshold': 0.3, 'multiplier': 2.2, 'power': 1.15},
            'intimate_couple': {'base_threshold': 0.35, 'multiplier': 2.8, 'power': 1.25},
            'dancing': {'base_threshold': 0.2, 'multiplier': 1.8, 'power': 1.05}
        }
        
        # IMPROVED EMOTION CALIBRATION PARAMETERS (removed surprised)
        self.emotion_calibration_params = {
            'happy': {'base_threshold': 0.25, 'multiplier': 2.2, 'power': 1.1},
            'sad': {'base_threshold': 0.3, 'multiplier': 2.0, 'power': 1.15},
            'angry': {'base_threshold': 0.35, 'multiplier': 2.3, 'power': 1.2},
            'fearful': {'base_threshold': 0.3, 'multiplier': 2.1, 'power': 1.1},
            'disgusted': {'base_threshold': 0.35, 'multiplier': 2.0, 'power': 1.15}
        }
        
        # Combine calibration parameters
        self.calibration_params = {**self.adult_calibration_params, **self.emotion_calibration_params}
        
        # PRECISION TIMING THRESHOLDS (HIGHER - more selective)
        self.precision_thresholds = {
            'kissing': 0.7, 'hugging': 0.65, 'nudity': 0.8, 'partial_nudity': 0.7,
            'intimate_couple': 0.8, 'dancing': 0.75,
            'happy': 0.75, 'sad': 0.7, 'angry': 0.6, 'fearful': 0.7, 'disgusted': 0.45
        }
        
        logger.info("Comprehensive detector V2 initialized: adult content + improved emotions + precision timing")
    
    def calibrate_score(self, raw_score: float, content_type: str) -> float:
        """Calibrate raw CLIP scores to proper 0-1 range with content-specific parameters."""
        params = self.calibration_params[content_type]
        
        # Apply content-specific threshold
        if raw_score < params['base_threshold']:
            return 0.0
        
        # Normalize above threshold
        normalized = (raw_score - params['base_threshold']) / (1.0 - params['base_threshold'])
        normalized = min(1.0, normalized)
        
        # Apply power transformation and multiplier
        calibrated = (normalized ** params['power']) * params['multiplier']
        
        # Final clipping to 0-1 range
        return min(1.0, max(0.0, calibrated))
    
    def detect_content_multi_method(self, frames: List[np.ndarray], content_type: str) -> float:
        """PROVEN multi-method detection from smart_selection_6s.py"""
        if not frames:
            return 0.0
        
        prompts_config = self.all_prompts[content_type]
        positive_prompts = prompts_config['positive']
        negative_prompts = prompts_config['negative']
        weight_factors = prompts_config['weight_factors']
        
        # Method 1: Individual prompt scoring
        individual_scores = []
        
        # Sample frames strategically
        num_samples = min(5, len(frames))
        if len(frames) > num_samples:
            indices = np.linspace(0, len(frames) - 1, num_samples, dtype=int)
            sample_frames = [frames[i] for i in indices]
        else:
            sample_frames = frames
        
        for frame in sample_frames:
            pil_image = Image.fromarray(frame)
            frame_scores = []
            
            # Test each positive prompt individually
            for i, prompt in enumerate(positive_prompts):
                # Compare positive vs negative
                all_prompts = [prompt] + negative_prompts
                
                inputs = self.clip_processor(
                    text=all_prompts,
                    images=pil_image,
                    return_tensors="pt",
                    padding=True
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.clip_model(**inputs)
                    logits = outputs.logits_per_image
                    probs = F.softmax(logits, dim=-1)
                
                # Score is positive prompt probability with weight factor
                positive_prob = probs[0][0].item()
                weighted_score = positive_prob * weight_factors[i]
                frame_scores.append(weighted_score)
            
            # Take maximum weighted score for this frame
            individual_scores.append(max(frame_scores))
        
        # Method 2: Ensemble prompt scoring
        ensemble_scores = []
        
        for frame in sample_frames:
            pil_image = Image.fromarray(frame)
            
            # All positive prompts vs all negative prompts
            all_prompts = positive_prompts + negative_prompts
            
            inputs = self.clip_processor(
                text=all_prompts,
                images=pil_image,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.clip_model(**inputs)
                logits = outputs.logits_per_image
                probs = F.softmax(logits, dim=-1)
            
            # Calculate weighted average of positive prompts
            positive_probs = probs[0][:len(positive_prompts)]
            weighted_positive = sum(p * w for p, w in zip(positive_probs, weight_factors))
            weighted_positive /= sum(weight_factors)  # Normalize by total weights
            
            ensemble_scores.append(weighted_positive.item())
        
        # Method 3: Comparative scoring (best vs worst)
        comparative_scores = []
        
        for frame in sample_frames:
            pil_image = Image.fromarray(frame)
            
            # Best positive vs best negative
            best_positive = positive_prompts[0]  # Highest weighted
            best_negative = negative_prompts[0]  # Most contrasting
            
            inputs = self.clip_processor(
                text=[best_positive, best_negative],
                images=pil_image,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.clip_model(**inputs)
                logits = outputs.logits_per_image
                probs = F.softmax(logits, dim=-1)
            
            # Relative confidence in positive vs negative
            pos_prob = probs[0][0].item()
            neg_prob = probs[0][1].item()
            relative_score = pos_prob / (pos_prob + neg_prob) if (pos_prob + neg_prob) > 0 else 0
            
            comparative_scores.append(relative_score)
        
        # Combine all methods with weights
        method_weights = [0.4, 0.4, 0.2]  # Individual, Ensemble, Comparative
        
        final_raw_score = (
            np.mean(individual_scores) * method_weights[0] +
            np.mean(ensemble_scores) * method_weights[1] +
            np.mean(comparative_scores) * method_weights[2]
        )
        
        # Apply content-specific calibration
        calibrated_score = self.calibrate_score(final_raw_score, content_type)
        
        logger.debug(f"{content_type} - Raw: {final_raw_score:.3f}, Calibrated: {calibrated_score:.3f}")
        
        return calibrated_score
    
    def extract_precise_frames(self, cap, timestamp: float, fps: float, duration: float = 1.0) -> List[np.ndarray]:
        """Extract frames for precise timing analysis (1-second window)"""
        start_time = max(0, timestamp - duration/2)
        end_time = timestamp + duration/2
        
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        
        # Extract 3-4 frames from 1-second window
        frame_indices = np.linspace(start_frame, end_frame - 1, 4, dtype=int)
        frames = []
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, (224, 224))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
        
        return frames
    
    def smart_frame_selection(self, cap, start_time: float, end_time: float, fps: float) -> List[np.ndarray]:
        """PROVEN two-stage smart frame selection from smart_selection_6s.py"""
        
        # Stage 1: Extract 4 frames per second (24 total for 6-second segment)
        segment_duration = end_time - start_time
        frames_per_second = 4
        total_candidate_frames = int(segment_duration * frames_per_second)
        
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        
        # Extract candidate frames evenly distributed
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
            return candidate_frames  # Return all if too few
        
        # Stage 2: Quick analysis - compute simple visual quality scores
        frame_scores = []
        for i, frame in enumerate(candidate_frames):
            # Simple quality metrics
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            
            # Variance of Laplacian (sharpness)
            sharpness = cv2.Laplacian(frame_gray, cv2.CV_64F).var()
            
            # Contrast (standard deviation)
            contrast = np.std(frame_gray)
            
            # Brightness balance (distance from 128)
            brightness_balance = 1.0 / (1.0 + abs(np.mean(frame_gray) - 128))
            
            # Combined quality score
            quality_score = sharpness * 0.5 + contrast * 0.3 + brightness_balance * 100 * 0.2
            
            frame_scores.append({
                'index': i,
                'frame': frame,
                'score': quality_score,
                'timestamp': start_time + (i / frames_per_second)
            })
        
        # Stage 3: Select best 6 frames
        # Sort by quality score and temporal distribution
        frame_scores.sort(key=lambda x: x['score'], reverse=True)
        
        # Take top candidates but ensure temporal distribution
        selected_frames = []
        selected_timestamps = []
        selected_indices = set()
        
        for frame_data in frame_scores:
            timestamp = frame_data['timestamp']
            frame_index = frame_data['index']
            
            # Check if this timestamp is well-distributed
            too_close = any(abs(timestamp - prev_ts) < 0.5 for prev_ts in selected_timestamps)
            
            if not too_close or len(selected_frames) < 3:  # Force at least 3 frames
                selected_frames.append(frame_data['frame'])
                selected_timestamps.append(timestamp)
                selected_indices.add(frame_index)
                
                if len(selected_frames) >= 6:
                    break
        
        # Fill remaining slots if needed
        while len(selected_frames) < 6 and len(selected_frames) < len(candidate_frames):
            for frame_data in frame_scores:
                if frame_data['index'] not in selected_indices:
                    selected_frames.append(frame_data['frame'])
                    selected_indices.add(frame_data['index'])
                    if len(selected_frames) >= 6:
                        break
        
        logger.debug(f"Smart selection: {total_candidate_frames} → {len(selected_frames)} frames")
        return selected_frames
    
    def extract_video_segments(self, video_path: str, start_time: float = 0, 
                              end_time: float = None, segment_duration: float = 6.0) -> List[Dict]:
        """PROVEN segment extraction from smart_selection_6s.py"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_duration = total_frames / fps
        
        if end_time is None:
            end_time = video_duration
        
        segments = []
        # NO overlap: segments are sequential 
        current_time = start_time
        
        while current_time < end_time:
            segment_end = min(current_time + segment_duration, end_time)
            
            # Two-stage smart frame selection
            smart_frames = self.smart_frame_selection(cap, current_time, segment_end, fps)
            
            if smart_frames:
                segments.append({
                    'start_time': current_time,
                    'end_time': segment_end,
                    'frames': smart_frames,
                    'fps': fps
                })
            
            current_time += segment_duration
        
        cap.release()
        logger.info(f"Extracted {len(segments)} non-overlapping video segments with smart frame selection")
        return segments
    
    def analyze_precise_timing(self, video_path: str, high_score_segments: List[Dict]) -> Dict:
        """NEW: Analyze precise timing for high-scoring segments"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {}
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        precise_detections = {}
        
        logger.info(f"Analyzing precise timing for {len(high_score_segments)} high-scoring segments...")
        
        for segment_info in tqdm(high_score_segments, desc="Precise timing analysis"):
            start_time = segment_info['start_time']
            end_time = segment_info['end_time']
            content_type = segment_info['content_type']
            segment_score = segment_info['score']
            
            # Analyze each second within the high-scoring segment
            second_scores = []
            for second_offset in range(int(end_time - start_time)):
                timestamp = start_time + second_offset
                
                # Extract precise frames for this second
                precise_frames = self.extract_precise_frames(cap, timestamp, fps)
                
                if precise_frames:
                    # Analyze this specific second
                    precise_score = self.detect_content_multi_method(precise_frames, content_type)
                    second_scores.append({
                        'timestamp': timestamp,
                        'score': precise_score
                    })
            
            # Find the peak second within this segment
            if second_scores:
                best_second = max(second_scores, key=lambda x: x['score'])
                if best_second['score'] > self.precision_thresholds[content_type]:
                    precise_detections[f"{content_type}_{start_time}"] = {
                        'content_type': content_type,
                        'segment_start': start_time,
                        'segment_end': end_time,
                        'segment_score': segment_score,
                        'precise_timestamp': best_second['timestamp'],
                        'precise_score': best_second['score'],
                        'all_second_scores': second_scores
                    }
        
        cap.release()
        return precise_detections
    
    def process_video(self, video_path: str, start_time: float = 0, 
                     end_time: float = None, segment_duration: float = 6.0) -> Tuple[List[Dict], Dict]:
        """Process video with comprehensive detection + precise timing for high scores"""
        
        logger.info("=== COMPREHENSIVE CONTENT & EMOTION DETECTION V2 ===")
        
        # Extract video segments
        segments = self.extract_video_segments(video_path, start_time, end_time, segment_duration)
        
        if not segments:
            return [], {}
        
        # All detection categories (removed surprised)
        adult_types = ['kissing', 'hugging', 'nudity', 'partial_nudity', 'intimate_couple', 'dancing']
        emotion_types = ['happy', 'sad', 'angry', 'fearful', 'disgusted']  # Removed surprised
        all_content_types = adult_types + emotion_types
        
        results = []
        high_score_segments = []
        
        # Initialize CSV file and batch processing
        os.makedirs("output", exist_ok=True)
        output_path = "output/comprehensive_detection_v9_blended_results.csv"
        batch_size = 3
        segment_batch_results = []
        csv_initialized = False
        
        # Process each segment
        for segment_idx, segment in enumerate(tqdm(segments, desc="Processing video segments")):
            frames = segment['frames']
            start_sec = segment['start_time']
            end_sec = segment['end_time']
            fps = segment['fps']
            
            # Detect all content types for this segment
            segment_scores = {}
            
            for content_type in all_content_types:
                score = self.detect_content_multi_method(frames, content_type)
                segment_scores[content_type] = score
                
                # Check if this segment has high scores for precise analysis
                if score > self.precision_thresholds[content_type]:
                    high_score_segments.append({
                        'start_time': start_sec,
                        'end_time': end_sec,
                        'content_type': content_type,
                        'score': score
                    })
            
            # Generate per-second predictions for this segment
            segment_duration_sec = end_sec - start_sec
            for second_offset in range(int(segment_duration_sec)):
                timestamp = start_sec + second_offset
                
                # Apply temporal decay (scores higher near segment center)
                segment_center = (start_sec + end_sec) / 2
                distance_from_center = abs(timestamp - segment_center)
                max_distance = segment_duration_sec / 2
                temporal_weight = 1.0 - (distance_from_center / max_distance) * 0.3  # Max 30% decay
                
                minutes = int(timestamp // 60)
                seconds = int(timestamp % 60)
                time_mmss = f"{minutes:02d}:{seconds:02d}"
                
                result = {
                    'time_mmss': time_mmss,
                    'second': timestamp,
                    'frame': int(timestamp * fps),
                    
                    # Adult content scores (PROVEN from smart_selection_6s.py)
                    'kissing_score': segment_scores['kissing'] * temporal_weight,
                    'hugging_score': segment_scores['hugging'] * temporal_weight,
                    'partial_nudity_score': segment_scores['partial_nudity'] * temporal_weight,
                    'intimate_couple_score': segment_scores['intimate_couple'] * temporal_weight,
                    'provocative_dancing_score': segment_scores['dancing'] * temporal_weight,
                    'nude_score': segment_scores['nudity'] * temporal_weight,
                    
                    # Emotion scores (IMPROVED - removed surprised)
                    'happy_score': segment_scores['happy'] * temporal_weight,
                    'sad_score': segment_scores['sad'] * temporal_weight,
                    'angry_score': segment_scores['angry'] * temporal_weight,
                    'fearful_score': segment_scores['fearful'] * temporal_weight,
                    'disgusted_score': segment_scores['disgusted'] * temporal_weight
                }
                results.append(result)
                segment_batch_results.append(result)
            
            # Save batch every 3 segments or at the end
            if (segment_idx + 1) % batch_size == 0 or segment_idx == len(segments) - 1:
                if segment_batch_results:
                    df_batch = pd.DataFrame(segment_batch_results)
                    
                    if not csv_initialized:
                        # First batch - create file with headers
                        df_batch.to_csv(output_path, index=False)
                        csv_initialized = True
                        logger.info(f"Initialized CSV and saved first batch (segments 0-{segment_idx})")
                    else:
                        # Subsequent batches - append without headers
                        df_batch.to_csv(output_path, mode='a', header=False, index=False)
                        logger.info(f"Saved batch: segments {max(0, segment_idx + 1 - batch_size)}-{segment_idx} ({len(segment_batch_results)} seconds)")
                    
                    segment_batch_results = []  # Clear batch
        
        logger.info(f"Generated predictions for {len(results)} seconds with comprehensive detection")
        
        # NEW: Precise timing analysis for high-scoring segments
        precise_detections = self.analyze_precise_timing(video_path, high_score_segments)
        
        # NEW: Create lookup for per-second scores from precise detections
        per_second_lookup = {}
        for detection_data in precise_detections.values():
            content_type = detection_data['content_type']
            for second_data in detection_data['all_second_scores']:
                timestamp = second_data['timestamp']
                score = second_data['score']
                if timestamp not in per_second_lookup:
                    per_second_lookup[timestamp] = {}
                per_second_lookup[timestamp][content_type] = score
        
        # NEW: Apply blending to existing results and update CSV file
        if per_second_lookup:  # Only if we have precise timing data
            logger.info("Applying blended scoring to CSV file...")
            
            # Read the current CSV file
            df_existing = pd.read_csv(output_path)
            
            # Apply blending to the DataFrame
            content_type_mapping = {
                'kissing_score': 'kissing',
                'hugging_score': 'hugging', 
                'partial_nudity_score': 'partial_nudity',
                'intimate_couple_score': 'intimate_couple',
                'provocative_dancing_score': 'dancing',
                'nude_score': 'nudity',
                'happy_score': 'happy',
                'sad_score': 'sad',
                'angry_score': 'angry',
                'fearful_score': 'fearful',
                'disgusted_score': 'disgusted'
            }
            
            blended_count = 0
            for idx, row in df_existing.iterrows():
                timestamp = row['second']
                per_second_scores = per_second_lookup.get(timestamp, {})
                
                # Blend scores where per-second data exists
                for csv_key, content_type in content_type_mapping.items():
                    if content_type in per_second_scores:
                        segment_score = row[csv_key]  # Current segment score with temporal weight
                        per_second_score = per_second_scores[content_type]
                        
                        # Blend: lambda * per_second + (1-lambda) * segment
                        blended_score = (
                            self.lambda_blend * per_second_score + 
                            (1 - self.lambda_blend) * segment_score
                        )
                        df_existing.at[idx, csv_key] = blended_score
                        blended_count += 1
            
            # Save the blended results back to CSV
            df_existing.to_csv(output_path, index=False)
            logger.info(f"Applied blending to {blended_count} score values and updated CSV")
            
            # Also apply blending to in-memory results for consistency
            for result in results:
                timestamp = result['second']
                per_second_scores = per_second_lookup.get(timestamp, {})
                
                for csv_key, content_type in content_type_mapping.items():
                    if content_type in per_second_scores:
                        segment_score = result[csv_key]
                        per_second_score = per_second_scores[content_type]
                        
                        blended_score = (
                            self.lambda_blend * per_second_score + 
                            (1 - self.lambda_blend) * segment_score
                        )
                        result[csv_key] = blended_score
        else:
            logger.info("No precise timing data available - CSV contains segment-based scores only")
        
        return results, precise_detections


def main():
    parser = argparse.ArgumentParser(description='Comprehensive Video Content & Emotion Detection V2 with Blended Scoring')
    parser.add_argument('--video_path', type=str, default='input/movie.mp4')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda', 'mps'])
    parser.add_argument('--start_time', type=float, default=114.0)
    parser.add_argument('--end_time', type=float, default=None)
    parser.add_argument('--segment_duration', type=float, default=6.0)
    parser.add_argument('--lambda_blend', type=float, default=0.75, 
                       help='Blend coefficient: lambda * per_second + (1-lambda) * segment (default: 0.75)')
    
    args = parser.parse_args()
    
    logger.info(f"Using device: {args.device}")
    logger.info(f"Lambda blend: {args.lambda_blend} (per-second: {args.lambda_blend:.1%}, segment: {1-args.lambda_blend:.1%})")
    
    try:
        # Initialize comprehensive detector V2 with blended scoring
        detector = ComprehensiveContentDetectorV2(device=args.device, lambda_blend=args.lambda_blend)
        
        # Process video
        results, precise_detections = detector.process_video(
            args.video_path,
            start_time=args.start_time,
            end_time=args.end_time,
            segment_duration=args.segment_duration
        )
        
        # Results already saved incrementally during processing
        # Final CSV file is ready with blended scores
        output_path = "output/comprehensive_detection_v9_blended_results.csv"
        logger.info(f"Final blended results available at {output_path}")
        
        # Save precise timing results
        if precise_detections:
            precise_path = "output/precise_timing_detections.json"
            with open(precise_path, 'w') as f:
                json.dump(precise_detections, f, indent=2)
            logger.info(f"Precise timing saved to {precise_path}")
        
        # Show score distribution
        print(f"\n=== COMPREHENSIVE DETECTION V2 BLENDED SCORE DISTRIBUTION ===")
        print(f"Lambda blend: {args.lambda_blend} (per-second: {args.lambda_blend:.1%}, segment: {1-args.lambda_blend:.1%})")
        
        # Read final CSV for statistics
        df = pd.read_csv(output_path)
        
        # Adult content scores
        print("\n--- ADULT CONTENT (BLENDED) ---")
        adult_cols = ['kissing_score', 'hugging_score', 'partial_nudity_score', 
                     'intimate_couple_score', 'provocative_dancing_score', 'nude_score']
        for col in adult_cols:
            scores = df[col].values
            print(f"{col}:")
            print(f"  Range: {scores.min():.3f} - {scores.max():.3f}")
            print(f"  Mean: {scores.mean():.3f}, Std: {scores.std():.3f}")
            print(f"  High confidence (>0.7): {(scores > 0.7).sum()}")
        
        # Emotion scores (improved - removed surprised)
        print("\n--- EMOTION DETECTION (BLENDED) ---")
        emotion_cols = ['happy_score', 'sad_score', 'angry_score', 
                       'fearful_score', 'disgusted_score']  # Removed surprised
        for col in emotion_cols:
            scores = df[col].values
            print(f"{col}:")
            print(f"  Range: {scores.min():.3f} - {scores.max():.3f}")
            print(f"  Mean: {scores.mean():.3f}, Std: {scores.std():.3f}")
            print(f"  High confidence (>0.7): {(scores > 0.7).sum()}")
        
        # Precise timing summary
        print(f"\n--- PRECISE TIMING ANALYSIS ---")
        if precise_detections:
            print(f"High-confidence precise detections: {len(precise_detections)}")
            for detection_id, details in precise_detections.items():
                timestamp = details['precise_timestamp']
                content_type = details['content_type']
                precise_score = details['precise_score']
                minutes = int(timestamp // 60)
                seconds = int(timestamp % 60)
                print(f"  {content_type}: {minutes:02d}:{seconds:02d} (score: {precise_score:.3f})")
        else:
            print("No high-confidence detections found for precise timing")
        
        print(f"\n✅ COMPREHENSIVE DETECTION V2 BLENDED SCORING COMPLETE")
        print(f"🔧 IMPROVEMENTS:")
        print(f"   • Removed surprised emotion (poor performance)")
        print(f"   • Added precise timing for high-confidence segments")
        print(f"   • BLENDED SCORING: {args.lambda_blend:.1%} per-second + {1-args.lambda_blend:.1%} segment")
        print(f"   • {len(precise_detections)} precise detections found")
        print(f"📊 Total seconds analyzed: {len(results)}")
        print(f"📁 Results: {output_path}")
        print(f"⏰ Precise timing: output/precise_timing_detections.json")
    
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
