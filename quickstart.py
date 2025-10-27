"""
Quick start script to test the voice assistant with minimal setup.

This script provides a simple way to test each component individually
or the complete pipeline.
"""

import sys
import argparse
from pathlib import Path


def check_dependencies():
    """Check if required dependencies are installed."""
    missing = []

    try:
        import torch
        if not torch.cuda.is_available():
            print("⚠️  Warning: CUDA not available. Models will run on CPU (slow).")
    except ImportError:
        missing.append("torch")

    try:
        import transformers
    except ImportError:
        missing.append("transformers")

    try:
        import TTS
    except ImportError:
        missing.append("TTS")

    if missing:
        print("❌ Missing dependencies:", ", ".join(missing))
        print("\nPlease install requirements:")
        print("  pip install -r requirements.txt")
        return False

    print("✓ All dependencies installed")
    return True


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Voice Assistant Quick Start")
    parser.add_argument(
        "mode",
        choices=["check", "asr", "llm", "tts", "full"],
        help="Mode to run: check dependencies, test individual services, or full pipeline"
    )
    parser.add_argument(
        "--audio",
        type=str,
        help="Path to audio file (for ASR and full mode)"
    )
    parser.add_argument(
        "--text",
        type=str,
        default="¿Cuál es la capital de Perú?",
        help="Text input (for LLM, TTS, and full mode)"
    )

    args = parser.parse_args()

    # Check dependencies
    if args.mode == "check":
        print("="*70)
        print("  Checking Dependencies")
        print("="*70)
        check_dependencies()

        try:
            import torch
            print(f"\nPyTorch version: {torch.__version__}")
            print(f"CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"CUDA version: {torch.version.cuda}")
                print(f"GPU: {torch.cuda.get_device_name(0)}")
                print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        except Exception as e:
            print(f"Error checking PyTorch: {e}")

        return

    # Check dependencies before running services
    if not check_dependencies():
        sys.exit(1)

    # Run selected mode
    if args.mode == "asr":
        from examples.test_asr import test_asr
        test_asr(args.audio)

    elif args.mode == "llm":
        from examples.test_llm import test_llm
        test_llm("4bit")

    elif args.mode == "tts":
        from examples.test_tts import test_tts
        test_tts("quickstart_output.wav")

    elif args.mode == "full":
        from examples.test_full_pipeline import test_full_pipeline
        test_full_pipeline(args.audio)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Voice Assistant Quick Start")
        print("="*70)
        print("\nUsage:")
        print("  python quickstart.py check              # Check dependencies")
        print("  python quickstart.py asr [--audio FILE] # Test ASR service")
        print("  python quickstart.py llm                # Test LLM service")
        print("  python quickstart.py tts                # Test TTS service")
        print("  python quickstart.py full [--audio FILE]# Test full pipeline")
        print("\nExamples:")
        print("  python quickstart.py check")
        print("  python quickstart.py llm")
        print("  python quickstart.py full --text 'Hola, ¿cómo estás?'")
        print("="*70)
        sys.exit(0)

    main()
