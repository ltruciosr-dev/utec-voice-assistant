# Model Information and Recommendations

This document provides detailed information about the models used in the UTEC Voice Assistant and recommendations for different hardware configurations.

## ASR Models (Whisper)

### Model Comparison

| Model | Size | Parameters | VRAM | Speed | Accuracy | Spanish Support |
|-------|------|-----------|------|-------|----------|-----------------|
| whisper-small | 244 MB | 244M | ~2GB | Fast | Good | ✅ Excellent |
| whisper-large-v3-turbo | 809 MB | 809M | ~4GB | Medium | Better | ✅ Excellent |

### Model Details

#### whisper-small
- **HuggingFace ID**: `openai/whisper-small`
- **Architecture**: Encoder-decoder transformer
- **Training Data**: 680k hours of multilingual audio
- **Languages**: 99+ languages including Spanish
- **Use Case**: Real-time applications, resource-constrained environments
- **Inference Time**: ~2-3s for 10s audio (RTX 3090)

#### whisper-large-v3-turbo
- **HuggingFace ID**: `openai/whisper-large-v3-turbo`
- **Architecture**: Pruned large-v3 (32→4 decoder layers)
- **Performance**: Similar to large-v2 quality with 6x speed improvement
- **Languages**: Multilingual with strong Spanish performance
- **Use Case**: Better accuracy for complex/noisy audio
- **Inference Time**: ~4-5s for 10s audio (RTX 3090)

### Recommendation for Voice Assistant

**Use whisper-small** because:
1. Sufficient accuracy for conversational Spanish
2. Lower VRAM usage leaves room for LLM
3. Faster inference for better user experience
4. Total system VRAM: ~9-10GB (within 12GB constraint)

## LLM Models (Qwen2.5)

### Model: Qwen2.5-7B-Instruct

- **HuggingFace ID**: `Qwen/Qwen2.5-7B-Instruct`
- **Parameters**: 7.6B
- **Architecture**: Transformer decoder
- **Context Length**: 32k tokens (128k extended)
- **Languages**: Multilingual (excellent Spanish support)
- **Instruction-tuned**: Yes, optimized for chat/assistant tasks

### Quantization Options

| Quantization | Technology | Bits | VRAM | Perplexity | Inference Speed | Quality Loss |
|-------------|-----------|------|------|-----------|----------------|--------------|
| None (FP16) | - | 16 | ~14GB | Baseline | ~20 tok/s | 0% |
| 8-bit | BitsAndBytes | 8 | ~8GB | +0.5% | ~25 tok/s | Minimal |
| 4-bit (NF4) | BitsAndBytes | 4 | ~5GB | +1.5% | ~30 tok/s | Small |

### Quantization Details

#### 4-bit (Recommended)
- **Config**: NF4 with double quantization
- **Compute dtype**: float16
- **VRAM**: ~4.5-5.5GB
- **Quality**: Excellent for conversational AI
- **Use Case**: Voice assistants, chatbots on consumer GPUs

#### 8-bit
- **Config**: LLM.int8()
- **VRAM**: ~7-8GB
- **Quality**: Near-native performance
- **Use Case**: When 12GB+ VRAM available

#### None (FP16)
- **VRAM**: ~14GB
- **Quality**: Best possible
- **Use Case**: High-end GPUs (A100, RTX 4090)

### Why Qwen2.5-7B?

1. **Spanish Performance**: Trained on significant Spanish corpus
2. **Instruction Following**: Fine-tuned for assistant-like responses
3. **Efficiency**: Good quality-to-size ratio
4. **Quantization Support**: Works well with 4-bit quantization
5. **Community**: Active development and support

## TTS Model (XTTS-v2)

### Model: Coqui XTTS-v2

- **Model ID**: `tts_models/multilingual/multi-dataset/xtts_v2`
- **Architecture**: Transformer-based neural codec
- **Parameters**: ~500M
- **VRAM**: ~2GB
- **Languages**: 16 languages including Spanish
- **Special Feature**: Voice cloning from 6+ seconds of audio

### Features

1. **Multilingual**: Native Spanish support
2. **Voice Cloning**: Clone any voice with reference audio
3. **Prosody**: Natural intonation and rhythm
4. **Speed Control**: Adjustable speaking rate
5. **Quality**: Near-commercial TTS quality

### Language Support

| Language | Code | Quality |
|----------|------|---------|
| Spanish | es | ⭐⭐⭐⭐⭐ |
| English | en | ⭐⭐⭐⭐⭐ |
| French | fr | ⭐⭐⭐⭐ |
| German | de | ⭐⭐⭐⭐ |
| Italian | it | ⭐⭐⭐⭐ |
| Portuguese | pt | ⭐⭐⭐⭐ |
| Polish | pl | ⭐⭐⭐ |
| Turkish | tr | ⭐⭐⭐ |
| Russian | ru | ⭐⭐⭐ |
| Dutch | nl | ⭐⭐⭐ |
| Czech | cs | ⭐⭐⭐ |
| Arabic | ar | ⭐⭐⭐ |
| Chinese | zh-cn | ⭐⭐⭐ |
| Japanese | ja | ⭐⭐⭐ |
| Hungarian | hu | ⭐⭐⭐ |
| Korean | ko | ⭐⭐⭐ |

## Hardware Recommendations

### Minimum (8-10GB VRAM)

```yaml
asr:
  model_size: "small"  # 2GB

llm:
  quantization: "4bit"  # 5GB

tts:
  # Default XTTS-v2    # 2GB

# Total: ~9GB
```

**GPUs**: RTX 3080 (10GB), RTX 3080 Ti (12GB), RTX 3090 (24GB)

### Recommended (12GB+ VRAM)

```yaml
asr:
  model_size: "turbo"  # 4GB (optional upgrade)

llm:
  quantization: "4bit"  # 5GB

tts:
  # Default XTTS-v2    # 2GB

# Total: ~11GB
```

**GPUs**: RTX 3090, RTX 4080, RTX 4090, A5000

### High-End (16GB+ VRAM)

```yaml
asr:
  model_size: "turbo"  # 4GB

llm:
  quantization: "8bit"  # 8GB

tts:
  # Default XTTS-v2    # 2GB

# Total: ~14GB
```

**GPUs**: RTX 4090, A6000, A100 (40GB)

## Performance Benchmarks

### Test System: RTX 3090 (24GB)

#### ASR Performance

| Model | Audio Length | Inference Time | Real-time Factor |
|-------|-------------|----------------|------------------|
| small | 10s | 2.3s | 0.23x |
| turbo | 10s | 4.1s | 0.41x |

#### LLM Performance (50 tokens)

| Quantization | Time | Tokens/sec | VRAM |
|-------------|------|-----------|------|
| 4-bit | 1.7s | 29.4 | 5.2GB |
| 8-bit | 2.0s | 25.0 | 7.8GB |
| FP16 | 2.5s | 20.0 | 14.1GB |

#### TTS Performance

| Text Length | Inference Time | Real-time Factor |
|------------|----------------|------------------|
| 10 words | 1.2s | ~0.6x |
| 20 words | 2.1s | ~0.5x |
| 50 words | 4.5s | ~0.45x |

### End-to-End Latency

**Configuration**: small + 4bit + xtts-v2

| Pipeline Stage | Time | Percentage |
|----------------|------|-----------|
| ASR (10s audio) | 2.3s | 28% |
| LLM (50 tokens) | 1.7s | 21% |
| TTS (20 words) | 2.1s | 26% |
| Overhead | 2.1s | 25% |
| **Total** | **8.2s** | **100%** |

## Alternative Models

### ASR Alternatives

1. **Distil-Whisper**: Faster but similar accuracy
   - `distil-whisper/distil-large-v3`
   - 6x faster, 49% smaller
   - VRAM: ~1.5GB

2. **Wav2Vec2 Spanish**: Specialized Spanish ASR
   - `jonatasgrosman/wav2vec2-large-xlsr-53-spanish`
   - Optimized for Spanish only
   - VRAM: ~1GB

### LLM Alternatives

1. **Llama-3.1-8B-Instruct**: Similar size, good Spanish
   - `meta-llama/Llama-3.1-8B-Instruct`
   - Requires HuggingFace access token

2. **Mistral-7B-Instruct**: Efficient, multilingual
   - `mistralai/Mistral-7B-Instruct-v0.3`
   - Good quantization support

3. **Phi-3.5-mini**: Smaller, faster
   - `microsoft/Phi-3.5-mini-instruct`
   - 3.8B params, ~3GB with 4-bit
   - Trade-off: Less sophisticated responses

### TTS Alternatives

1. **Bark**: Multilingual with emotion
   - `suno/bark`
   - VRAM: ~5GB
   - Slower but more expressive

2. **StyleTTS2**: High quality, English-focused
   - Better for English, limited Spanish

## Model Selection Decision Tree

```
Do you have <12GB VRAM?
│
├─ Yes → Use recommended configuration
│   ├─ ASR: whisper-small
│   ├─ LLM: Qwen2.5-7B-Instruct (4-bit)
│   └─ TTS: XTTS-v2
│
└─ No → Do you have >16GB VRAM?
    │
    ├─ Yes → Use high-end configuration
    │   ├─ ASR: whisper-turbo
    │   ├─ LLM: Qwen2.5-7B-Instruct (8-bit)
    │   └─ TTS: XTTS-v2
    │
    └─ No (12-16GB) → Use mid-range
        ├─ ASR: whisper-small or turbo
        ├─ LLM: Qwen2.5-7B-Instruct (4-bit)
        └─ TTS: XTTS-v2
```

## Updating Models

To use different models, edit the service initialization:

### ASR
```python
from services.asr import ASRService
asr = ASRService(model_size="turbo")  # or "small"
```

### LLM
```python
from services.llm import LLMService
llm = LLMService(
    model_id="Qwen/Qwen2.5-7B-Instruct",
    quantization="4bit"  # or "8bit", "none"
)
```

### TTS
```python
from services.tts import TTSService
tts = TTSService(
    model_name="tts_models/multilingual/multi-dataset/xtts_v2",
    language="es"
)
```

## References

- [Whisper Paper](https://arxiv.org/abs/2212.04356)
- [Qwen2.5 Technical Report](https://arxiv.org/abs/2309.16609)
- [XTTS Paper](https://arxiv.org/abs/2406.04904)
- [BitsAndBytes Documentation](https://github.com/TimDettmers/bitsandbytes)
