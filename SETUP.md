# Setup and Installation Guide

This guide will help you set up the UTEC Voice Assistant on your machine.

## Prerequisites

- **Python 3.9, 3.10, or 3.11** (Required: TTS library does not support Python 3.12+)
- CUDA-capable NVIDIA GPU with at least 12GB VRAM
- CUDA Toolkit 11.8 or higher
- ~30GB free disk space for models

**Important**: If you have Python 3.12 or newer, you'll need to create a Python 3.11 environment:
```bash
# Using conda
conda create -n voice-assistant python=3.11
conda activate voice-assistant

# Or using pyenv
pyenv install 3.11.9
pyenv local 3.11.9
```

## Installation Steps

### 1. Clone the Repository

```bash
git clone <repository-url>
cd utec-voice-assistant
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### 3. Install PyTorch with CUDA Support

Install PyTorch with CUDA support first (required for GPU acceleration):

```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: If you encounter issues with `bitsandbytes` on Windows, you may need to install a Windows-compatible version:

```bash
pip install bitsandbytes-windows
```

### 5. Download Models

The models will be downloaded automatically on first use, but you can pre-download them:

```python
# Run this to pre-download all models
python -c "
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, AutoModelForCausalLM, AutoTokenizer
from TTS.api import TTS

# Download ASR model
print('Downloading ASR model...')
AutoModelForSpeechSeq2Seq.from_pretrained('openai/whisper-small')
AutoProcessor.from_pretrained('openai/whisper-small')

# Download LLM model (this will take a while)
print('Downloading LLM model...')
AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-7B-Instruct')
AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct')

# Download TTS model
print('Downloading TTS model...')
TTS('tts_models/multilingual/multi-dataset/xtts_v2')

print('All models downloaded!')
"
```

### 6. Verify Installation

Test each service independently:

```bash
# Test ASR service
python examples/test_asr.py

# Test LLM service
python examples/test_llm.py

# Test TTS service
python examples/test_tts.py
```

## Configuration

Edit `config/config.yaml` to customize the voice assistant behavior:

```yaml
# ASR Configuration
asr:
  model_size: "small"  # Options: "small" or "turbo"
  language: "spanish"

# LLM Configuration
llm:
  quantization: "4bit"  # Options: "4bit", "8bit", or "none"
  temperature: 0.7

# TTS Configuration
tts:
  language: "es"
  speed: 1.0
```

## VRAM Requirements

### Recommended Configuration (< 12GB VRAM)

- **ASR**: Whisper Small (~2GB)
- **LLM**: Qwen2.5-7B-Instruct with 4-bit quantization (~5GB)
- **TTS**: XTTS-v2 (~2GB)
- **Total**: ~9-10GB

### Alternative Configurations

#### If you have more VRAM (12GB+)

```yaml
asr:
  model_size: "turbo"  # Better accuracy

llm:
  quantization: "8bit"  # Better quality
```

#### If you have less VRAM

```yaml
asr:
  model_size: "small"

llm:
  quantization: "4bit"  # Keep at 4-bit

# Consider running services separately or on CPU
```

## Usage

### Interactive Mode

```bash
python voice_assistant.py
```

### Process Audio File

```bash
python voice_assistant.py --audio input.wav --output response.wav
```

### Text Chat (Bypass ASR)

```bash
python voice_assistant.py --text "¿Cuál es la capital de Perú?" --output response.wav
```

### Show System Information

```bash
python voice_assistant.py --info
```

## Troubleshooting

### Out of Memory Error

If you encounter CUDA out of memory errors:

1. Reduce batch size in ASR configuration
2. Use 4-bit quantization for LLM
3. Close other applications using GPU memory
4. Run services separately instead of all at once

### Slow Performance

1. Ensure you're using GPU (check with `torch.cuda.is_available()`)
2. Use smaller models (whisper-small instead of turbo)
3. Reduce `max_new_tokens` in LLM configuration

### Import Errors

If you get import errors:

```bash
# Reinstall dependencies
pip install --upgrade --force-reinstall -r requirements.txt
```

### BitsAndBytes Issues on Windows

```bash
# Install Windows-compatible version
pip uninstall bitsandbytes
pip install bitsandbytes-windows
```

## Performance Benchmarks

### ASR Service

| Model | Parameters | Inference Time* | VRAM |
|-------|-----------|----------------|------|
| whisper-small | 244M | ~2-3s | ~2GB |
| whisper-large-v3-turbo | 809M | ~4-5s | ~4GB |

*For 10-second audio on RTX 3090

### LLM Service

| Quantization | VRAM | Inference Speed** | Quality |
|-------------|------|------------------|---------|
| 4-bit | ~5GB | ~30 tokens/s | Good |
| 8-bit | ~8GB | ~25 tokens/s | Better |
| None (fp16) | ~14GB | ~20 tokens/s | Best |

**For Qwen2.5-7B-Instruct on RTX 3090

### TTS Service

- Inference Time: ~2-3s for 20 words
- VRAM: ~2GB
- Quality: High (comparable to commercial TTS)

## Next Steps

1. Try the comparison script for ASR models:
   ```bash
   python services/asr/compare_models.py
   ```

2. Test the full pipeline:
   ```bash
   python examples/test_full_pipeline.py
   ```

3. Customize the system prompt in `config/config.yaml` for your use case

4. Add voice cloning by providing a speaker reference:
   ```yaml
   tts:
     speaker_wav: "path/to/speaker_reference.wav"
   ```

## Support

For issues or questions:
- Check the [README.md](README.md) for architecture overview
- Review the example scripts in `examples/`
- Check configuration in `config/config.yaml`
