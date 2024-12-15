import torch
from transformers import AutoTokenizer, AutoProcessor, LlavaForConditionalGeneration
from peft import PeftModel, PeftConfig

def load_finetuned_model(base_model_path: str, adapter_path: str, device: str = "cuda"):
    """
    Load a fine-tuned model with LoRA adapters.
    
    Args:
        base_model_path: Path to the base model or huggingface model id
        adapter_path: Path to the trained LoRA adapter weights
        device: Device to load the model on ("cuda" or "cpu")
    
    Returns:
        model: The loaded model with adapters
        processor: The processor for handling both text and images
    """
    # Load the base model
    model = LlavaForConditionalGeneration.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto" if device == "cuda" else None,
        low_cpu_mem_usage=True,
    )
    
    # Load processor from original model
    processor = AutoProcessor.from_pretrained(base_model_path)
    
    # Load and apply LoRA adapters
    model = PeftModel.from_pretrained(
        model,
        adapter_path,
        device_map="auto" if device == "cuda" else None
    )
    
    # Merge LoRA weights with base model if needed
    model = model.merge_and_unload()
    
    if device == "cpu":
        model = model.to("cpu")
    
    return model, processor

# Example usage
if __name__ == "__main__":
    # Replace these paths with your actual paths
    BASE_MODEL_PATH = "llava-hf/llava-1.5-7b-hf"  # or HF model id
    ADAPTER_PATH = "checkpoints/llava-1.5-7b_lora-True_qlora-False"  # where you saved the fine-tuned model
    
    # Load the model
    model, processor = load_finetuned_model(
        base_model_path=BASE_MODEL_PATH,
        adapter_path=ADAPTER_PATH,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Example inference
    inputs = processor(text="Hello, how are you?", return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=50)
    response = processor.decode(outputs[0], skip_special_tokens=True)
    print(response)