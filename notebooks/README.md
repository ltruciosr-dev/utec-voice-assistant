# Voice Assistant Notebooks

This directory contains Jupyter notebooks for testing and evaluating each component of the voice assistant system.

## Notebooks Overview

### 1. ASR Testing (`01_asr_testing.ipynb`)
**Automatic Speech Recognition evaluation**

Features:
- Microphone recording functionality
- Test three ASR models:
  - Whisper Small (OpenAI)
  - Whisper Large v3 Turbo (OpenAI)
  - Parakeet TDT 0.6B v3 (NVIDIA)
- Performance comparison (speed, accuracy, memory)
- Batch testing with multiple recordings
- Model recommendations

**Usage:**
```python
# Record audio and test all models
audio_file = record_audio(duration=5, output_dir="recordings")
result_small = transcribe_with_whisper(audio_file, whisper_small_pipe, "Whisper Small")
result_turbo = transcribe_with_whisper(audio_file, whisper_turbo_pipe, "Whisper Turbo")
result_parakeet = transcribe_with_parakeet(audio_file, parakeet_pipe, "Parakeet TDT")
```

### 2. LLM Testing (`02_llm_testing.ipynb`)
**Language Model evaluation with Spanish focus**

Features:
- Test Qwen2.5-7B-Instruct with different quantizations
- GPU memory monitoring
- Spanish language comprehension tests
- Performance metrics (tokens/sec, latency)
- Quality evaluation across question types
- Quantization comparison (4-bit, 8-bit, FP16)

**Test Categories:**
- General knowledge
- Technology explanations
- Mathematics
- History
- Science
- Problem solving
- Creative writing
- Complex reasoning

**Usage:**
```python
# Load and test 4-bit model
model_4bit, tokenizer_4bit = load_llm_4bit()
result = generate_response(model_4bit, tokenizer_4bit, "¿Cuál es la capital de Perú?")
```

### 3. TTS Voice Cloning (`03_tts_voice_cloning.ipynb`)
**Text-to-Speech with zero-shot voice cloning**

Features:
- Voice cloning with XTTS-v2 (no fine-tuning needed!)
- Record speaker reference audio
- Multilingual synthesis
- Speaking speed control
- Performance analysis
- Best practices guide

**Voice Cloning Steps:**
1. Record 10+ seconds of reference audio
2. Use reference for any text synthesis
3. Clone voice across multiple languages

**Usage:**
```python
# Record reference
speaker_ref = record_speaker_reference(duration=10)

# Synthesize with cloned voice
result = synthesize_with_cloning(
    text="Hola, ¿cómo estás?",
    speaker_wav=speaker_ref,
    output_path="output.wav",
    language="es"
)
```

### 4. Full Pipeline Integration (`04_full_pipeline_integration.ipynb`)
**End-to-end voice assistant testing**

Features:
- Complete pipeline: Audio → ASR → LLM → TTS → Audio
- Detailed timing for each stage
- GPU memory tracking
- Bottleneck identification
- Multiple test runs for averaging
- Performance recommendations

**Pipeline Metrics:**
- ASR latency
- LLM inference time
- TTS synthesis time
- Total end-to-end latency
- Memory usage per stage
- Real-time factor

**Usage:**
```python
# Run complete pipeline
result = run_full_pipeline(
    audio_input="test.wav",
    speaker_reference="my_voice.wav",
    test_name="Test 1"
)
```

## Getting Started

### 1. Install Dependencies

```bash
pip install -r ../requirements.txt
```

### 2. Install Jupyter

```bash
pip install jupyter notebook
# Or for JupyterLab
pip install jupyterlab
```

### 3. Launch Jupyter

```bash
# From notebooks directory
jupyter notebook

# Or JupyterLab
jupyter lab
```

### 4. Run Notebooks in Order

Start with notebook 01 and progress through 04:
1. `01_asr_testing.ipynb` - Test ASR models
2. `02_llm_testing.ipynb` - Test LLM
3. `03_tts_voice_cloning.ipynb` - Test TTS with voice cloning
4. `04_full_pipeline_integration.ipynb` - Test complete system

## Hardware Requirements

### Minimum
- GPU: 8GB VRAM (e.g., RTX 3070)
- RAM: 16GB system memory
- Storage: 30GB free space for models

### Recommended
- GPU: 12GB+ VRAM (e.g., RTX 3080 Ti, RTX 3090)
- RAM: 32GB system memory
- Storage: 50GB free space

### Expected VRAM Usage

| Notebook | Models Loaded | Approx VRAM |
|----------|---------------|-------------|
| 01_asr | All 3 ASR models (sequentially) | ~4GB peak |
| 02_llm | Qwen2.5-7B (4-bit) | ~5GB |
| 03_tts | XTTS-v2 | ~2GB |
| 04_full | ASR + LLM + TTS | ~9-10GB |

## Tips for Best Results

### ASR (Notebook 01)
- Use clear, uncompressed audio (WAV format)
- Minimize background noise when recording
- Speak naturally at normal pace
- Test with 5-10 second audio clips

### LLM (Notebook 02)
- Use 4-bit quantization for <12GB VRAM
- Allow ~30s for first generation (model compilation)
- Subsequent generations are much faster
- Adjust `max_new_tokens` for response length

### TTS (Notebook 03)
- Record 10-15 seconds of reference audio for voice cloning
- Speak complete sentences in clear voice
- Use quiet environment for reference recording
- Reference audio quality affects output quality

### Full Pipeline (Notebook 04)
- Run multiple tests for average performance
- Allow GPU to warm up (first run may be slower)
- Monitor memory usage throughout
- Clear GPU cache between major operations

## Troubleshooting

### Out of Memory Errors

```python
# Clear GPU cache
import torch
torch.cuda.empty_cache()

# Or restart kernel and load models one at a time
```

### Microphone Not Working

```python
# List available audio devices
import sounddevice as sd
print(sd.query_devices())

# Set default device
sd.default.device = 0  # Change number based on query_devices output
```

### Model Download Failures

```python
# Set HuggingFace cache directory
import os
os.environ['HF_HOME'] = '/path/to/large/storage'
os.environ['TRANSFORMERS_CACHE'] = '/path/to/large/storage'
```

### Slow Performance

1. Verify GPU is being used:
   ```python
   import torch
   print(torch.cuda.is_available())
   print(torch.cuda.get_device_name(0))
   ```

2. Check CUDA version compatibility
3. Close other GPU-intensive applications
4. Reduce batch sizes or token limits

## Expected Results

### ASR Performance
- Whisper Small: ~2-3s for 10s audio
- Whisper Turbo: ~4-5s for 10s audio
- Parakeet TDT: ~1-2s for 10s audio

### LLM Performance
- 4-bit quantization: ~25-30 tokens/s
- Response time: 2-5s for 50 tokens
- First generation slower (compilation)

### TTS Performance
- Synthesis: ~2-3s for 20 words
- Real-time factor: <1.0x (faster than real-time)
- Voice cloning: Works with 6+ second reference

### Full Pipeline
- End-to-end: 8-15s total
- ASR: ~25-30% of time
- LLM: ~20-25% of time
- TTS: ~20-30% of time
- Overhead: ~20-25% of time

## Additional Resources

- [Whisper Documentation](https://github.com/openai/whisper)
- [Qwen2.5 Model Card](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
- [XTTS-v2 Documentation](https://docs.coqui.ai/en/latest/models/xtts.html)
- [BitsAndBytes Quantization](https://github.com/TimDettmers/bitsandbytes)

## Contributing

Found a bug or have suggestions? Please update the notebooks or create a pull request!

## License

MIT License - See main project LICENSE file
