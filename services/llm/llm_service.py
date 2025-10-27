"""
LLM Service using Qwen2.5-7B-Instruct with quantization support.

Supports multiple quantization levels to fit within VRAM constraints:
- 4-bit: ~4-5 GB VRAM (recommended for <12GB total)
- 8-bit: ~7-8 GB VRAM
- float16: ~14 GB VRAM (not recommended for this use case)
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from typing import Optional, List, Dict, Union
import warnings


class LLMService:
    """
    Language Model service using Qwen2.5-7B-Instruct.

    Attributes:
        model_id (str): HuggingFace model identifier
        device (str): Device to run the model on
        quantization (str): Quantization level ('4bit', '8bit', or 'none')
        model: The loaded language model
        tokenizer: The model's tokenizer
    """

    DEFAULT_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

    def __init__(
        self,
        model_id: str = None,
        quantization: str = "4bit",
        device: Optional[str] = None,
        max_memory: Optional[dict] = None
    ):
        """
        Initialize the LLM service.

        Args:
            model_id: HuggingFace model ID (default: Qwen/Qwen2.5-7B-Instruct)
            quantization: Quantization level ('4bit', '8bit', or 'none')
            device: Device to use ('cuda' or 'cpu'). Auto-detected if None.
            max_memory: Maximum memory per device (dict)
        """
        self.model_id = model_id or self.DEFAULT_MODEL_ID
        self.quantization = quantization.lower()

        if self.quantization not in ["4bit", "8bit", "none"]:
            raise ValueError("quantization must be '4bit', '8bit', or 'none'")

        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        if self.quantization != "none" and self.device == "cpu":
            warnings.warn("Quantization is only supported on CUDA. Falling back to no quantization.")
            self.quantization = "none"

        print(f"Initializing LLM service with model: {self.model_id}")
        print(f"Device: {self.device}, Quantization: {self.quantization}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=True
        )

        # Configure quantization
        model_kwargs = {
            "trust_remote_code": True,
            "device_map": "auto",
        }

        if self.quantization == "4bit":
            print("Using 4-bit quantization (BitsAndBytes)")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            model_kwargs["quantization_config"] = quantization_config

        elif self.quantization == "8bit":
            print("Using 8-bit quantization (BitsAndBytes)")
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
            model_kwargs["quantization_config"] = quantization_config

        else:
            print("Using float16 precision (no quantization)")
            model_kwargs["torch_dtype"] = torch.float16

        if max_memory:
            model_kwargs["max_memory"] = max_memory

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            **model_kwargs
        )

        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"LLM service initialized successfully!")

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
        repetition_penalty: float = 1.1,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Generate text response from a prompt.

        Args:
            prompt: User input text
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            do_sample: Whether to use sampling (vs greedy decoding)
            repetition_penalty: Penalty for repeating tokens
            system_prompt: Optional system prompt to set context

        Returns:
            Generated text response
        """
        # Format messages for Qwen chat template
        messages = []

        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })

        messages.append({
            "role": "user",
            "content": prompt
        })

        # Apply chat template
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        ).to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
                repetition_penalty=repetition_penalty,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only the generated tokens (skip input)
        generated_ids = outputs[0][inputs.input_ids.shape[1]:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return response.strip()

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """
        Generate response from a conversation history.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional generation parameters

        Returns:
            Generated text response
        """
        # Apply chat template
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096
        ).to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )

        # Decode
        generated_ids = outputs[0][inputs.input_ids.shape[1]:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return response.strip()

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
            "model_id": self.model_id,
            "device": self.device,
            "quantization": self.quantization,
            "parameters": sum(p.numel() for p in self.model.parameters()) / 1e9,
            "memory_mb": memory_mb,
            "vocab_size": len(self.tokenizer)
        }


if __name__ == "__main__":
    # Example usage
    print("LLM Service - Example Usage\n")

    # Initialize with 4-bit quantization (recommended)
    llm = LLMService(quantization="4bit")

    # Get model info
    info = llm.get_model_info()
    print(f"\nModel Info:")
    for key, value in info.items():
        print(f"  {key}: {value}")

    # Test generation
    print("\n" + "="*60)
    print("Testing Spanish language understanding...")
    print("="*60)

    system_prompt = "Eres un asistente de voz útil y amigable. Responde de manera concisa y clara."
    prompt = "¿Cuál es la capital de Perú?"

    print(f"\nUser: {prompt}")
    response = llm.generate(prompt, system_prompt=system_prompt, max_new_tokens=100)
    print(f"Assistant: {response}")

    print("\nLLM service is ready!")
