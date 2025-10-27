"""
Voice Assistant Integration Script

This script integrates all three services (ASR, LLM, TTS) into a complete voice assistant.
"""

import os
import sys
import yaml
import argparse
from pathlib import Path
from typing import Optional, List, Dict

# Add services to path
sys.path.append(str(Path(__file__).parent))

from services.asr import ASRService
from services.llm import LLMService
from services.tts import TTSService


class VoiceAssistant:
    """
    Complete voice assistant combining ASR, LLM, and TTS services.

    Flow:
        Audio Input → ASR → Text → LLM → Response Text → TTS → Audio Output
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the voice assistant with all three services.

        Args:
            config_path: Path to configuration YAML file
        """
        print("="*70)
        print("  UTEC Voice Assistant - Initializing")
        print("="*70)

        # Load configuration
        self.config = self._load_config(config_path)

        # Initialize conversation history
        self.conversation_history: List[Dict[str, str]] = []
        self.max_history = self.config.get("assistant", {}).get("max_conversation_history", 10)

        # Initialize services
        self._initialize_services()

        print("\n" + "="*70)
        print("  Voice Assistant Ready!")
        print("="*70)

    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        if not os.path.exists(config_path):
            print(f"⚠️  Config file not found: {config_path}")
            print("Using default configuration...")
            return self._get_default_config()

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        print(f"✓ Configuration loaded from: {config_path}")
        return config

    def _get_default_config(self) -> dict:
        """Return default configuration."""
        return {
            "asr": {
                "model_size": "small",
                "language": "spanish",
                "compute_type": "float16"
            },
            "llm": {
                "model_id": "Qwen/Qwen2.5-7B-Instruct",
                "quantization": "4bit",
                "max_new_tokens": 512,
                "temperature": 0.7,
                "system_prompt": "Eres un asistente de voz útil y amigable. Responde de manera concisa y clara."
            },
            "tts": {
                "model_name": "tts_models/multilingual/multi-dataset/xtts_v2",
                "language": "es",
                "speed": 1.0
            },
            "assistant": {
                "max_conversation_history": 10
            }
        }

    def _initialize_services(self):
        """Initialize ASR, LLM, and TTS services."""
        print("\n[1/3] Initializing ASR Service...")
        asr_config = self.config.get("asr", {})
        self.asr = ASRService(
            model_size=asr_config.get("model_size", "small"),
            language=asr_config.get("language", "spanish"),
            compute_type=asr_config.get("compute_type", "float16")
        )

        print("\n[2/3] Initializing LLM Service...")
        llm_config = self.config.get("llm", {})
        self.llm = LLMService(
            model_id=llm_config.get("model_id"),
            quantization=llm_config.get("quantization", "4bit")
        )

        print("\n[3/3] Initializing TTS Service...")
        tts_config = self.config.get("tts", {})
        self.tts = TTSService(
            model_name=tts_config.get("model_name"),
            language=tts_config.get("language", "es"),
            speaker_wav=tts_config.get("speaker_wav")
        )

        # Store generation parameters
        self.llm_params = {
            "max_new_tokens": llm_config.get("max_new_tokens", 512),
            "temperature": llm_config.get("temperature", 0.7),
            "top_p": llm_config.get("top_p", 0.9),
            "top_k": llm_config.get("top_k", 50),
            "repetition_penalty": llm_config.get("repetition_penalty", 1.1),
            "system_prompt": llm_config.get("system_prompt")
        }

        self.tts_params = {
            "speed": tts_config.get("speed", 1.0)
        }

    def process_audio(
        self,
        audio_path: str,
        output_path: Optional[str] = None,
        verbose: bool = True
    ) -> Dict[str, str]:
        """
        Process audio input through the complete pipeline.

        Args:
            audio_path: Path to input audio file
            output_path: Path to save TTS output (optional)
            verbose: Print intermediate results

        Returns:
            Dictionary with transcription, response, and output_path
        """
        if verbose:
            print("\n" + "="*70)
            print("  Processing Audio Input")
            print("="*70)

        # Step 1: ASR - Speech to Text
        if verbose:
            print("\n[1/3] 🎤 Transcribing audio...")
        transcription_result = self.asr.transcribe(audio_path)
        transcription = transcription_result["text"]

        if verbose:
            print(f"User: {transcription}")

        # Step 2: LLM - Generate Response
        if verbose:
            print("\n[2/3] 🤖 Generating response...")

        response = self.llm.generate(
            prompt=transcription,
            **self.llm_params
        )

        if verbose:
            print(f"Assistant: {response}")

        # Update conversation history
        self._add_to_history("user", transcription)
        self._add_to_history("assistant", response)

        # Step 3: TTS - Text to Speech
        if verbose:
            print("\n[3/3] 🔊 Synthesizing speech...")

        if output_path is None:
            output_path = "output_response.wav"

        self.tts.synthesize(
            text=response,
            output_path=output_path,
            **self.tts_params
        )

        if verbose:
            print(f"✓ Audio response saved to: {output_path}")
            print("\n" + "="*70)

        return {
            "transcription": transcription,
            "response": response,
            "output_path": output_path
        }

    def chat(
        self,
        user_input: str,
        output_path: Optional[str] = None,
        verbose: bool = True
    ) -> Dict[str, str]:
        """
        Chat with text input (bypassing ASR).

        Args:
            user_input: Text input from user
            output_path: Path to save TTS output (optional)
            verbose: Print results

        Returns:
            Dictionary with response and output_path
        """
        if verbose:
            print(f"\nUser: {user_input}")

        # Generate response
        response = self.llm.generate(
            prompt=user_input,
            **self.llm_params
        )

        if verbose:
            print(f"Assistant: {response}")

        # Update history
        self._add_to_history("user", user_input)
        self._add_to_history("assistant", response)

        # Synthesize speech
        if output_path is None:
            output_path = "output_response.wav"

        self.tts.synthesize(
            text=response,
            output_path=output_path,
            **self.tts_params
        )

        if verbose:
            print(f"✓ Audio saved to: {output_path}")

        return {
            "response": response,
            "output_path": output_path
        }

    def _add_to_history(self, role: str, content: str):
        """Add message to conversation history."""
        self.conversation_history.append({
            "role": role,
            "content": content
        })

        # Trim history if too long
        if len(self.conversation_history) > self.max_history * 2:  # *2 for user+assistant pairs
            self.conversation_history = self.conversation_history[-self.max_history * 2:]

    def get_system_info(self) -> dict:
        """Get information about all loaded models."""
        return {
            "asr": self.asr.get_model_info(),
            "llm": self.llm.get_model_info(),
            "tts": self.tts.get_model_info()
        }

    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []
        print("✓ Conversation history cleared")


def main():
    """Main function for CLI usage."""
    parser = argparse.ArgumentParser(description="UTEC Voice Assistant")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--audio",
        type=str,
        help="Path to input audio file"
    )
    parser.add_argument(
        "--text",
        type=str,
        help="Text input (bypasses ASR)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output.wav",
        help="Path to save output audio"
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Show system information and exit"
    )

    args = parser.parse_args()

    # Initialize assistant
    assistant = VoiceAssistant(config_path=args.config)

    # Show info and exit
    if args.info:
        print("\n" + "="*70)
        print("  System Information")
        print("="*70)
        info = assistant.get_system_info()
        for service, data in info.items():
            print(f"\n{service.upper()}:")
            for key, value in data.items():
                print(f"  {key}: {value}")
        return

    # Process audio input
    if args.audio:
        result = assistant.process_audio(args.audio, args.output)

    # Process text input
    elif args.text:
        result = assistant.chat(args.text, args.output)

    # Interactive mode
    else:
        print("\n" + "="*70)
        print("  Interactive Mode")
        print("="*70)
        print("Commands:")
        print("  - Enter text to chat")
        print("  - Type 'exit' or 'quit' to end")
        print("  - Type 'clear' to clear conversation history")
        print("  - Type 'info' to show system information")
        print("="*70)

        while True:
            try:
                user_input = input("\nYou: ").strip()

                if user_input.lower() in ["exit", "quit"]:
                    print("\nGoodbye!")
                    break

                if user_input.lower() == "clear":
                    assistant.clear_history()
                    continue

                if user_input.lower() == "info":
                    info = assistant.get_system_info()
                    for service, data in info.items():
                        print(f"\n{service.upper()}:")
                        for key, value in data.items():
                            print(f"  {key}: {value}")
                    continue

                if not user_input:
                    continue

                result = assistant.chat(user_input, verbose=False)
                print(f"Assistant: {result['response']}")

            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")


if __name__ == "__main__":
    main()
