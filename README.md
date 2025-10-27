# UTEC Voice Assistant

A modular voice assistant system composed of three independent services:

1. **ASR (Automatic Speech Recognition)**: Converts speech to text using Whisper models
2. **LLM (Large Language Model)**: Processes text and generates responses using Qwen2.5-7B-Instruct
3. **TTS (Text-to-Speech)**: Converts text responses to speech using XTTS-v2

## Architecture

```
User Audio Input → ASR Service → Text → LLM Service → Response Text → TTS Service → Audio Output
```

## Models Used

### ASR Service
- **Small Model**: `openai/whisper-small` (244M parameters)
- **Medium/Turbo Model**: `openai/whisper-large-v3-turbo` (809M parameters)
- Both models support Spanish and multilingual speech recognition

### LLM Service
- **Model**: `Qwen/Qwen2.5-7B-Instruct`
- **Quantization**: 4-bit or 8-bit (to fit within 12GB VRAM constraint)
- Supports Spanish language understanding and generation

### TTS Service
- **Model**: `coqui/XTTS-v2`
- Multi-lingual text-to-speech with voice cloning capabilities

## Project Structure

```
utec-voice-assistant/
├── services/
│   ├── asr/          # Speech recognition service
│   ├── llm/          # Language model service
│   └── tts/          # Text-to-speech service
├── config/           # Configuration files
├── examples/         # Usage examples
├── tests/            # Test files
└── requirements.txt  # Python dependencies
```

## Requirements

- Python 3.9+
- CUDA-capable GPU with <12GB VRAM
- PyTorch with CUDA support

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd utec-voice-assistant

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Notebooks

**Interactive Jupyter notebooks** for testing each component:

- `01_asr_testing.ipynb` - ASR model comparison with microphone recording
- `02_llm_testing.ipynb` - LLM evaluation with Spanish questions
- `03_tts_voice_cloning.ipynb` - TTS with zero-shot voice cloning
- `04_full_pipeline_integration.ipynb` - Complete pipeline testing

See [notebooks/README.md](notebooks/README.md) for detailed usage instructions.

## Quick Start

### 1. Check Dependencies

```bash
python quickstart.py check
```

### 2. Test Individual Services

```bash
# Test ASR (Speech Recognition)
python quickstart.py asr

# Test LLM (Language Model)
python quickstart.py llm

# Test TTS (Text-to-Speech)
python quickstart.py tts
```

### 3. Test Full Pipeline

```bash
# Text input
python quickstart.py full --text "¿Cuál es la capital de Perú?"

# Audio input (requires audio file)
python quickstart.py full --audio input.wav
```

## Usage Examples

### Interactive Chat

```bash
python voice_assistant.py
```

### Process Audio File

```bash
python voice_assistant.py --audio input.wav --output response.wav
```

### Text-only Mode

```bash
python voice_assistant.py --text "Explícame qué es la inteligencia artificial"
```

### Show System Information

```bash
python voice_assistant.py --info
```

## Model Comparison

### ASR Models

Run the comparison script to evaluate different Whisper models:

```bash
cd services/asr
python compare_models.py [audio_file.wav]
```

This will compare:
- **whisper-small** (244M params, ~2GB VRAM)
- **whisper-large-v3-turbo** (809M params, ~4GB VRAM)

**Recommendation**: Use `small` for voice assistant to save VRAM for LLM.

### LLM Quantization Options

The LLM service supports three quantization levels:

| Quantization | VRAM | Quality | Speed |
|-------------|------|---------|-------|
| 4-bit | ~5GB | Good | Fast |
| 8-bit | ~8GB | Better | Medium |
| None (fp16) | ~14GB | Best | Slower |

**Recommendation**: Use `4-bit` for <12GB VRAM systems.

## Configuration

Edit `config/config.yaml` to customize behavior:

```yaml
asr:
  model_size: "small"  # "small" or "turbo"
  language: "spanish"

llm:
  quantization: "4bit"  # "4bit", "8bit", or "none"
  temperature: 0.7
  max_new_tokens: 512
  system_prompt: "Your custom system prompt"

tts:
  language: "es"
  speed: 1.0
  speaker_wav: null  # Add path for voice cloning
```

## Memory Usage

Total VRAM estimation for recommended configuration:

- ASR (small): ~2GB
- LLM (4-bit): ~5GB
- TTS (XTTS-v2): ~2GB
- **Total**: ~9-10GB

## Example Scripts

All examples are in the `examples/` directory:

- `test_asr.py` - Test ASR service independently
- `test_llm.py` - Test LLM service with different quantizations
- `test_tts.py` - Test TTS service
- `test_full_pipeline.py` - Test complete voice assistant

## Features

- ✅ Multilingual support (Spanish optimized)
- ✅ Voice cloning with XTTS-v2
- ✅ Memory-efficient 4-bit quantization
- ✅ Conversation history management
- ✅ Configurable system prompts
- ✅ Real-time audio processing
- ✅ Modular service architecture

## Troubleshooting

### Out of Memory

```yaml
# Use smaller models
asr:
  model_size: "small"
llm:
  quantization: "4bit"
```

### Slow Performance

1. Ensure CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`
2. Use GPU-optimized models
3. Reduce `max_new_tokens` in config

### Import Errors

```bash
pip install --upgrade -r requirements.txt
```

See [SETUP.md](SETUP.md) for detailed troubleshooting.

## Project Structure

```
utec-voice-assistant/
├── services/
│   ├── asr/
│   │   ├── asr_service.py       # ASR implementation
│   │   └── compare_models.py    # Model comparison script
│   ├── llm/
│   │   └── llm_service.py       # LLM with quantization
│   └── tts/
│       └── tts_service.py       # TTS implementation
├── config/
│   └── config.yaml              # Configuration file
├── notebooks/                   # Jupyter notebooks for testing
│   ├── 01_asr_testing.ipynb
│   ├── 02_llm_testing.ipynb
│   ├── 03_tts_voice_cloning.ipynb
│   ├── 04_full_pipeline_integration.ipynb
│   └── README.md
├── examples/
│   ├── test_asr.py
│   ├── test_llm.py
│   ├── test_tts.py
│   └── test_full_pipeline.py
├── voice_assistant.py           # Main integration script
├── quickstart.py                # Quick start utility
├── requirements.txt
├── SETUP.md                     # Detailed setup guide
├── MODEL_INFO.md                # Detailed model information
└── README.md
```

## Contributing

This is an educational project for UTEC. Contributions are welcome!

## License

MIT License
