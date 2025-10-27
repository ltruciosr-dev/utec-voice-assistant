"""
Script to compare Whisper small vs turbo models for Spanish ASR.

This script helps evaluate:
- Inference speed
- Model size
- VRAM usage
- Transcription quality (manual evaluation)
"""

import torch
from asr_service import ASRService
import time
import sys


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60 + "\n")


def get_memory_usage():
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2
    return 0


def compare_models(audio_path: str = None):
    """
    Compare small and turbo Whisper models.

    Args:
        audio_path: Optional path to audio file for testing
    """
    print_section("Whisper Model Comparison for Spanish ASR")

    models = ["small", "turbo"]
    results = {}

    for model_size in models:
        print(f"\n{'='*60}")
        print(f"Testing: {model_size.upper()} model")
        print(f"{'='*60}\n")

        try:
            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

            initial_memory = get_memory_usage()

            # Initialize model
            print(f"Loading {model_size} model...")
            start_load = time.time()
            asr = ASRService(model_size=model_size, language="spanish")
            load_time = time.time() - start_load

            model_memory = get_memory_usage() - initial_memory
            peak_memory = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0

            # Get model info
            info = asr.get_model_info()

            results[model_size] = {
                "model_id": info["model_id"],
                "parameters": info["parameters"],
                "load_time": load_time,
                "memory_mb": model_memory,
                "peak_memory_mb": peak_memory,
                "device": info["device"],
                "dtype": info["dtype"]
            }

            print(f"\n📊 Model Statistics:")
            print(f"   Model ID: {info['model_id']}")
            print(f"   Parameters: {info['parameters']:.2f}M")
            print(f"   Load Time: {load_time:.2f}s")
            print(f"   Memory Usage: {model_memory:.2f} MB")
            print(f"   Peak Memory: {peak_memory:.2f} MB")
            print(f"   Device: {info['device']}")
            print(f"   Data Type: {info['dtype']}")

            # If audio path provided, benchmark transcription
            if audio_path:
                print(f"\n🎤 Transcription Test:")
                benchmark = asr.benchmark(audio_path)
                print(f"   Inference Time: {benchmark['inference_time']:.2f}s")
                print(f"   Transcription: {benchmark['transcription']}")
                print(f"   Text Length: {benchmark['text_length']} chars")

                results[model_size]["inference_time"] = benchmark["inference_time"]
                results[model_size]["transcription"] = benchmark["transcription"]

            # Cleanup
            del asr
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"❌ Error testing {model_size} model: {str(e)}")
            results[model_size] = {"error": str(e)}

    # Print comparison summary
    print_section("Comparison Summary")

    if len(results) >= 2:
        print(f"{'Metric':<25} {'Small':<25} {'Turbo':<25}")
        print("-" * 75)

        metrics = [
            ("Parameters (M)", "parameters", "{:.2f}"),
            ("Load Time (s)", "load_time", "{:.2f}"),
            ("Memory Usage (MB)", "memory_mb", "{:.2f}"),
            ("Peak Memory (MB)", "peak_memory_mb", "{:.2f}"),
        ]

        if audio_path:
            metrics.append(("Inference Time (s)", "inference_time", "{:.2f}"))

        for label, key, fmt in metrics:
            small_val = results.get("small", {}).get(key, "N/A")
            turbo_val = results.get("turbo", {}).get(key, "N/A")

            if isinstance(small_val, (int, float)) and isinstance(turbo_val, (int, float)):
                small_str = fmt.format(small_val)
                turbo_str = fmt.format(turbo_val)
            else:
                small_str = str(small_val)
                turbo_str = str(turbo_val)

            print(f"{label:<25} {small_str:<25} {turbo_str:<25}")

        # Print recommendations
        print_section("Recommendations")

        print("🔹 Small Model (openai/whisper-small):")
        print("   - Faster inference")
        print("   - Lower memory footprint (~1-2 GB)")
        print("   - Good for real-time applications")
        print("   - Suitable when speed is priority")

        print("\n🔹 Turbo Model (openai/whisper-large-v3-turbo):")
        print("   - Higher accuracy (large-v2 level)")
        print("   - Better for complex/noisy audio")
        print("   - Moderate memory usage (~3-4 GB)")
        print("   - 6x faster than full large-v3")

        print("\n💡 For voice assistant with <12GB VRAM constraint:")
        print("   Recommended: SMALL model")
        print("   Reasoning: Leaves more VRAM for LLM (Qwen2.5-7B-Instruct)")
        print("   Total estimated: ~2GB (ASR) + ~8GB (LLM-4bit) + ~2GB (TTS) = ~12GB")


if __name__ == "__main__":
    audio_file = sys.argv[1] if len(sys.argv) > 1 else None

    if audio_file:
        print(f"Testing with audio file: {audio_file}")
    else:
        print("No audio file provided. Running model loading comparison only.")
        print("Usage: python compare_models.py <path_to_audio.wav>")

    compare_models(audio_file)
