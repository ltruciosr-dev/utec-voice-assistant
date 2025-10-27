import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from typing import Optional, Union
import numpy as np


class ASRService:
    """
    Automatic Speech Recognition service using models from HuggingFace.

    Attributes:
        model_size (str): Either 'small' or 'turbo'
        device (str): Device to run the model on ('cuda' or 'cpu')
        model: The loaded Whisper model
        processor: The model's processor
        pipe: The ASR pipeline
    """

    MODEL_MAPPING = {
        "small": "openai/whisper-small",
        "turbo": "openai/whisper-large-v3-turbo"
    }

    def __init__(
        self,
        model_size: str = "small",
        device: Optional[str] = None,
        language: str = "spanish",
        compute_type: str = "float16"
    ):
        """
        Initialize the ASR service.

        Args:
            model_size: Size of the model ('small' or 'turbo')
            device: Device to use ('cuda' or 'cpu'). Auto-detected if None.
            language: Target language for transcription (default: 'spanish')
            compute_type: Compute dtype ('float16' or 'float32')
        """
        if model_size not in self.MODEL_MAPPING:
            raise ValueError(f"Model size must be one of {list(self.MODEL_MAPPING.keys())}")

        self.model_size = model_size
        self.model_id = self.MODEL_MAPPING[model_size]
        self.language = language

        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Set compute type
        self.torch_dtype = torch.float16 if compute_type == "float16" and self.device == "cuda" else torch.float32

        print(f"Initializing ASR service with model: {self.model_id}")
        print(f"Device: {self.device}, Compute type: {self.torch_dtype}")

        # Load model
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_id,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True
        )
        self.model.to(self.device)

        # Load processor
        self.processor = AutoProcessor.from_pretrained(self.model_id)

        # Create pipeline
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            torch_dtype=self.torch_dtype,
            device=self.device,
        )

        print(f"ASR service initialized successfully!")

    def transcribe(
        self,
        audio: Union[str, np.ndarray],
        return_timestamps: bool = False,
        chunk_length_s: int = 30,
        batch_size: int = 16
    ) -> dict:
        """
        Transcribe audio to text.

        Args:
            audio: Path to audio file or numpy array of audio samples
            return_timestamps: Whether to return word-level timestamps
            chunk_length_s: Length of audio chunks for processing
            batch_size: Batch size for processing

        Returns:
            Dictionary containing 'text' and optionally 'chunks' with timestamps
        """
        generate_kwargs = {
            "language": self.language,
            "task": "transcribe"
        }

        result = self.pipe(
            audio,
            generate_kwargs=generate_kwargs,
            return_timestamps=return_timestamps,
            chunk_length_s=chunk_length_s,
            batch_size=batch_size
        )

        return result

    def get_model_info(self) -> dict:
        """
        Get information about the loaded model.

        Returns:
            Dictionary with model information
        """
        return {
            "model_size": self.model_size,
            "model_id": self.model_id,
            "device": self.device,
            "language": self.language,
            "dtype": str(self.torch_dtype),
            "parameters": sum(p.numel() for p in self.model.parameters()) / 1e6
        }

    def benchmark(self, audio_path: str) -> dict:
        """
        Benchmark the model on a given audio file.

        Args:
            audio_path: Path to audio file

        Returns:
            Dictionary with benchmark results (time, text length, etc.)
        """
        import time

        start_time = time.time()
        result = self.transcribe(audio_path)
        end_time = time.time()

        return {
            "transcription": result["text"],
            "inference_time": end_time - start_time,
            "text_length": len(result["text"]),
            "model_info": self.get_model_info()
        }


if __name__ == "__main__":
    # Example usage
    print("ASR Service - Example Usage\n")

    # Initialize with small model
    asr = ASRService(model_size="small")

    # Get model info
    info = asr.get_model_info()
    print(f"\nModel Info:")
    for key, value in info.items():
        print(f"  {key}: {value}")

    print("\nASR service is ready to transcribe audio files!")
    print("Usage: result = asr.transcribe('path/to/audio.wav')")
