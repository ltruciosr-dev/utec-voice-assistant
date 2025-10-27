"""
TTS (Text-to-Speech) Service using Coqui XTTS-v2.

XTTS-v2 is a multilingual TTS model that supports:
- Multiple languages including Spanish
- Voice cloning from a reference audio
- High-quality speech synthesis
"""

import torch
from TTS.api import TTS
from typing import Optional, Union
import os
import numpy as np


class TTSService:
    """
    Text-to-Speech service using Coqui XTTS-v2.

    Attributes:
        device (str): Device to run the model on ('cuda' or 'cpu')
        model_name (str): TTS model identifier
        tts: The TTS model instance
        speaker_wav (str): Path to reference speaker audio for voice cloning
    """

    DEFAULT_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"

    def __init__(
        self,
        model_name: str = None,
        device: Optional[str] = None,
        speaker_wav: Optional[str] = None,
        language: str = "es"
    ):
        """
        Initialize the TTS service.

        Args:
            model_name: TTS model name (default: XTTS-v2)
            device: Device to use ('cuda' or 'cpu'). Auto-detected if None.
            speaker_wav: Path to reference speaker audio for voice cloning
            language: Target language code (default: 'es' for Spanish)
        """
        self.model_name = model_name or self.DEFAULT_MODEL
        self.language = language
        self.speaker_wav = speaker_wav

        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Initializing TTS service with model: {self.model_name}")
        print(f"Device: {self.device}, Language: {self.language}")

        # Initialize TTS
        self.tts = TTS(model_name=self.model_name).to(self.device)

        print(f"TTS service initialized successfully!")

        # Check if speaker wav exists
        if self.speaker_wav and not os.path.exists(self.speaker_wav):
            print(f"⚠️  Warning: Speaker audio file not found: {self.speaker_wav}")
            print("   Using default speaker voice.")
            self.speaker_wav = None

    def synthesize(
        self,
        text: str,
        output_path: Optional[str] = None,
        speaker_wav: Optional[str] = None,
        language: Optional[str] = None,
        speed: float = 1.0
    ) -> Optional[np.ndarray]:
        """
        Synthesize speech from text.

        Args:
            text: Input text to synthesize
            output_path: Path to save audio file (WAV format). If None, returns numpy array
            speaker_wav: Path to reference speaker audio (overrides instance default)
            language: Language code (overrides instance default)
            speed: Speech speed multiplier (1.0 = normal)

        Returns:
            Numpy array of audio samples if output_path is None, otherwise None
        """
        # Use provided parameters or fall back to instance defaults
        speaker = speaker_wav or self.speaker_wav
        lang = language or self.language

        # Check if we need speaker for cloning
        if speaker is None:
            print("⚠️  No speaker audio provided. Using default voice.")
            print("   For voice cloning, provide a speaker_wav parameter.")

        # Generate speech
        if output_path:
            # Save to file
            self.tts.tts_to_file(
                text=text,
                file_path=output_path,
                speaker_wav=speaker,
                language=lang,
                speed=speed
            )
            print(f"✓ Audio saved to: {output_path}")
            return None
        else:
            # Return numpy array
            wav = self.tts.tts(
                text=text,
                speaker_wav=speaker,
                language=lang,
                speed=speed
            )
            return np.array(wav)

    def synthesize_with_emotion(
        self,
        text: str,
        output_path: Optional[str] = None,
        emotion: str = "neutral",
        **kwargs
    ) -> Optional[np.ndarray]:
        """
        Synthesize speech with specified emotion/style.

        Note: XTTS-v2 emotion control is primarily through speaker reference.
        This is a wrapper that may use different speaker samples for different emotions.

        Args:
            text: Input text
            output_path: Output file path
            emotion: Emotion/style ('neutral', 'happy', 'sad', etc.)
            **kwargs: Additional parameters for synthesize()

        Returns:
            Audio as numpy array or None if saved to file
        """
        # In XTTS-v2, emotion is controlled through speaker reference
        # You could map emotions to different speaker samples here
        print(f"Synthesizing with emotion: {emotion}")
        print("Note: Emotion control in XTTS-v2 is primarily through speaker reference.")

        return self.synthesize(text=text, output_path=output_path, **kwargs)

    def set_speaker(self, speaker_wav: str):
        """
        Set the reference speaker audio for voice cloning.

        Args:
            speaker_wav: Path to reference speaker audio
        """
        if not os.path.exists(speaker_wav):
            raise FileNotFoundError(f"Speaker audio file not found: {speaker_wav}")

        self.speaker_wav = speaker_wav
        print(f"✓ Speaker set to: {speaker_wav}")

    def get_model_info(self) -> dict:
        """
        Get information about the loaded model.

        Returns:
            Dictionary with model information
        """
        memory_mb = 0
        if torch.cuda.is_available():
            memory_mb = torch.cuda.memory_allocated() / 1024**2

        return {
            "model_name": self.model_name,
            "device": self.device,
            "language": self.language,
            "speaker_wav": self.speaker_wav,
            "memory_mb": memory_mb,
            "supported_languages": self.get_supported_languages()
        }

    def get_supported_languages(self) -> list:
        """
        Get list of supported languages for XTTS-v2.

        Returns:
            List of language codes
        """
        # XTTS-v2 supports these languages
        return [
            "en",  # English
            "es",  # Spanish
            "fr",  # French
            "de",  # German
            "it",  # Italian
            "pt",  # Portuguese
            "pl",  # Polish
            "tr",  # Turkish
            "ru",  # Russian
            "nl",  # Dutch
            "cs",  # Czech
            "ar",  # Arabic
            "zh-cn",  # Chinese (Simplified)
            "ja",  # Japanese
            "hu",  # Hungarian
            "ko"   # Korean
        ]

    def list_models(self) -> list:
        """
        List available TTS models.

        Returns:
            List of available model names
        """
        return TTS().list_models()


if __name__ == "__main__":
    # Example usage
    print("TTS Service - Example Usage\n")

    # Initialize TTS service
    tts = TTSService(language="es")

    # Get model info
    info = tts.get_model_info()
    print(f"\nModel Info:")
    for key, value in info.items():
        if key != "supported_languages":
            print(f"  {key}: {value}")

    print(f"\nSupported Languages: {', '.join(info['supported_languages'])}")

    # Example synthesis
    print("\n" + "="*60)
    print("Example: Synthesizing Spanish text")
    print("="*60)

    text = "Hola, soy un asistente de voz creado con inteligencia artificial."
    print(f"\nText: {text}")
    print("To generate audio: tts.synthesize(text, output_path='output.wav')")

    print("\nTTS service is ready!")
    print("\nFor voice cloning, provide a speaker_wav parameter:")
    print("  tts.synthesize(text, speaker_wav='path/to/speaker.wav', output_path='output.wav')")
