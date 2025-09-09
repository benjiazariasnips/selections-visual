# Comprehensive Video Content & Emotion Detection

A state-of-the-art Python pipeline for detecting adult content and emotions in videos using CLIP (Contrastive Language-Image Pre-training) with advanced multi-method detection and precise timing analysis.

## Features

### Adult Content Detection
- **Kissing**: Romantic lip-to-lip contact detection
- **Hugging**: Physical embrace and affectionate contact
- **Nudity**: Complete exposure detection
- **Partial Nudity**: Underwear, topless, or revealing clothing
- **Intimate Couple**: Sexual or erotic positioning between people
- **Provocative Dancing**: Suggestive or sensual dance movements

### Emotion Detection
- **Happy**: Joy, laughter, and positive expressions
- **Sad**: Sorrow, crying, and melancholic expressions
- **Angry**: Rage, fury, and hostile expressions
- **Fearful**: Fear, anxiety, and scared expressions
- **Disgusted**: Revulsion, distaste, and aversion expressions

### Advanced Features
- **Multi-Method Detection**: Combines individual, ensemble, and comparative scoring
- **Smart Frame Selection**: Intelligent quality-based frame sampling
- **Precise Timing Analysis**: Second-level analysis for high-confidence segments
- **Blended Scoring**: Combines segment-level and per-second predictions
- **Temporal Weighting**: Higher scores near segment centers
- **Batch Processing**: Efficient incremental CSV output

## Installation

### Option 1: Docker (Recommended)

**Quick Start with Docker:**

```bash
# Clone the repository
git clone https://github.com/benjiazariasnips/selections-visual.git
cd selections-visual

# Place your video file in input/ directory
cp your_video.mp4 input/

# Run with Docker (auto-detects GPU)
./run_docker.sh input/your_video.mp4
```

**Docker Commands:**

```bash
# Build the image
docker build -t selections-visual .

# Run with GPU support (if available)
docker run --rm --gpus all \
  -v $(pwd)/input:/app/input:ro \
  -v $(pwd)/output:/app/output \
  selections-visual python comprehensive_detection_v9.py --video_path input/movie.mp4

# Run CPU-only
docker run --rm \
  -v $(pwd)/input:/app/input:ro \
  -v $(pwd)/output:/app/output \
  selections-visual python comprehensive_detection_v9.py --video_path input/movie.mp4 --device cpu
```

**Docker Compose:**

```bash
# Run with docker-compose (GPU)
docker-compose up video-detection

# Run CPU-only version
docker-compose --profile cpu-only up video-detection-cpu
```

### Option 2: Local Installation

1. **Clone the repository**:
```bash
git clone https://github.com/benjiazariasnips/selections-visual.git
cd selections-visual
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

### Requirements

The system requires:
- **PyTorch**: Deep learning framework
- **Transformers**: For CLIP model access
- **OpenCV**: Video processing
- **PIL/Pillow**: Image processing
- **NumPy/Pandas**: Data manipulation
- **scikit-learn**: Machine learning utilities
- **tqdm**: Progress bars

## Usage

### Basic Usage

Process a video with default settings:

```bash
python comprehensive_detection_v9.py --video_path input/movie.mp4
```

### Advanced Usage

```bash
python comprehensive_detection_v9.py \
    --video_path input/movie.mp4 \
    --device auto \
    --start_time 0 \
    --end_time 300 \
    --segment_duration 6.0 \
    --lambda_blend 0.75
```

### Parameters

- `--video_path`: Path to input video file (default: `input/movie.mp4`)
- `--device`: Computing device (`auto`, `cpu`, `cuda`, `mps`) (default: `auto`)
- `--start_time`: Start time in seconds (default: `114.0`)
- `--end_time`: End time in seconds (default: full video)
- `--segment_duration`: Segment length in seconds (default: `6.0`)
- `--lambda_blend`: Blending coefficient for precise timing (default: `0.75`)

### Device Selection

The system automatically detects the best available device:
- **CUDA**: NVIDIA GPUs with CUDA support
- **MPS**: Apple Silicon (M1/M2/M3) GPUs
- **CPU**: Fallback for systems without GPU acceleration

## Output

### CSV Results

The main output is `output/comprehensive_detection_v9_blended_results.csv` with columns:

| Column | Description |
|--------|-------------|
| `time_mmss` | Timestamp in MM:SS format |
| `second` | Timestamp in seconds |
| `frame` | Frame number |
| `kissing_score` | Kissing detection confidence (0-1) |
| `hugging_score` | Hugging detection confidence (0-1) |
| `partial_nudity_score` | Partial nudity confidence (0-1) |
| `intimate_couple_score` | Intimate couple confidence (0-1) |
| `provocative_dancing_score` | Dancing confidence (0-1) |
| `nude_score` | Nudity confidence (0-1) |
| `happy_score` | Happy emotion confidence (0-1) |
| `sad_score` | Sad emotion confidence (0-1) |
| `angry_score` | Angry emotion confidence (0-1) |
| `fearful_score` | Fearful emotion confidence (0-1) |
| `disgusted_score` | Disgusted emotion confidence (0-1) |

### Precise Timing Results

High-confidence detections are saved to `output/precise_timing_detections.json` with exact timestamps and detailed analysis.

### Sample Output

```csv
time_mmss,second,frame,kissing_score,hugging_score,partial_nudity_score,...
01:54,114,2736,0.127,0.234,0.089,...
01:55,115,2760,0.156,0.267,0.098,...
01:56,116,2784,0.891,0.445,0.123,...
```

## Technical Details

### Detection Pipeline

1. **Segment Extraction**: Video divided into 6-second segments
2. **Smart Frame Selection**: Quality-based sampling (4 FPS → 6 best frames)
3. **Multi-Method Analysis**: 
   - Individual prompt scoring
   - Ensemble prompt comparison
   - Comparative positive/negative analysis
4. **Calibration**: Content-specific score normalization
5. **Precise Timing**: Second-level analysis for high scores
6. **Blended Scoring**: Combines segment and per-second predictions

### CLIP-Based Detection

Uses OpenAI's CLIP model with carefully crafted prompts:
- **Positive prompts**: Detailed descriptions of target content
- **Negative prompts**: Contrasting descriptions for calibration
- **Weight factors**: Importance ranking for multiple prompts
- **Calibration parameters**: Content-specific score adjustment

### Blended Scoring Formula

```
final_score = λ × per_second_score + (1-λ) × segment_score
```

Where λ (lambda_blend) defaults to 0.75, giving 75% weight to precise per-second analysis and 25% to segment-level context.

## Performance

### Processing Speed
- **GPU (CUDA)**: ~15-20 segments/minute
- **Apple Silicon (MPS)**: ~10-15 segments/minute  
- **CPU**: ~3-5 segments/minute

### Memory Requirements
- **GPU Memory**: ~4-6GB VRAM
- **System RAM**: ~8-12GB recommended
- **Storage**: ~2GB for models + output space

### Accuracy Characteristics
- **Adult Content**: High precision on clear, well-lit scenes
- **Emotions**: Good performance on frontal faces with clear expressions
- **Temporal Context**: Improved accuracy through segment-based analysis
- **Calibration**: Reduces false positives through multi-method scoring

## Configuration

### Detection Thresholds

Precision timing thresholds (in `comprehensive_detection_v9.py`):

```python
precision_thresholds = {
    'kissing': 0.7, 'hugging': 0.65, 'nudity': 0.8, 
    'partial_nudity': 0.7, 'intimate_couple': 0.8, 'dancing': 0.75,
    'happy': 0.75, 'sad': 0.7, 'angry': 0.6, 
    'fearful': 0.7, 'disgusted': 0.45
}
```

### Calibration Parameters

Each content type has specific calibration settings for optimal score normalization.

## Troubleshooting

### Common Issues

1. **GPU Memory Error**:
   ```bash
   # Use CPU or reduce segment duration
   python comprehensive_detection_v9.py --device cpu
   
   # With Docker
   ./run_docker.sh input/movie.mp4 --device cpu --segment_duration 8.0
   ```

2. **Slow Processing**:
   ```bash
   # Increase segment duration or reduce video length
   python comprehensive_detection_v9.py --segment_duration 8.0 --end_time 120
   
   # With Docker
   ./run_docker.sh input/movie.mp4 --segment_duration 8.0 --end_time 120
   ```

3. **Low Confidence Scores**:
   - Check video quality and lighting
   - Ensure faces/content are clearly visible
   - Consider adjusting calibration parameters

4. **CLIP Model Download**:
   ```bash
   # Ensure internet connection for first run
   # Model will be cached locally (~1.7GB)
   ```

### Docker-Specific Issues

1. **GPU Not Detected in Docker**:
   ```bash
   # Install nvidia-docker2
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   sudo apt-get update && sudo apt-get install -y nvidia-docker2
   sudo systemctl restart docker
   ```

2. **Permission Denied on output/**:
   ```bash
   # Fix ownership of output directory
   sudo chown -R $(id -u):$(id -g) output/
   ```

3. **Video File Not Found**:
   ```bash
   # Ensure video is in input/ directory
   ls -la input/
   # Video files are mounted read-only to /app/input in container
   ```

4. **Container Out of Memory**:
   ```bash
   # Increase Docker memory limit
   docker run --memory=8g --memory-swap=8g ...
   
   # Or modify docker-compose.yml mem_limit
   ```

### Debug Information

Enable detailed logging by checking the console output which includes:
- Device detection and model loading
- Segment processing progress
- Score distributions
- Precise timing analysis results

## Example Results

### High-Confidence Detections
```
=== PRECISE TIMING ANALYSIS ===
High-confidence precise detections: 3
  kissing: 02:15 (score: 0.892)
  hugging: 03:42 (score: 0.734)
  happy: 01:58 (score: 0.821)
```

### Score Distribution
```
--- ADULT CONTENT (BLENDED) ---
kissing_score:
  Range: 0.000 - 0.892
  Mean: 0.156, Std: 0.223
  High confidence (>0.7): 12

--- EMOTION DETECTION (BLENDED) ---
happy_score:
  Range: 0.000 - 0.821
  Mean: 0.234, Std: 0.187
  High confidence (>0.7): 8
```

## Privacy and Ethics

⚠️ **Important Considerations:**

- **Content Moderation**: Designed for safety and compliance applications
- **Privacy Compliance**: Ensure proper consent and legal compliance
- **Bias Awareness**: CLIP models may have dataset-specific biases
- **Responsible Use**: Use ethically and in accordance with local laws
- **Data Security**: Handle sensitive content detection results securely

## Contributing

Contributions welcome! Areas for improvement:
- Additional emotion categories
- Cultural bias reduction
- Performance optimization
- New content detection types

## License

MIT License - See LICENSE file for details.

## Acknowledgments

- **OpenAI CLIP**: Foundation model for multimodal understanding
- **PyTorch**: Deep learning framework
- **Transformers**: Model access and utilities
- **OpenCV**: Video processing capabilities

## Support

For issues:
1. Check troubleshooting section
2. Review console output for errors
3. Create GitHub issue with:
   - System specifications
   - Error messages
   - Video characteristics
   - Expected vs actual behavior

---

**Disclaimer**: This tool is intended for legitimate content analysis and moderation. Users are responsible for ethical and legal use in compliance with applicable regulations.