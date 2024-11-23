import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import torch.nn.functional as F

def load_models():
    # Load the draft model (small LoRA fine-tuned model)
    draft_model_path = "./qwen2.5_0.5B_lora_gsm8k"
    base_draft_model_path = "/home/asperger/DPSD/models/Qwen2.5-0.5B-Instruct"
    draft_tokenizer = AutoTokenizer.from_pretrained(base_draft_model_path, trust_remote_code=True)
    base_draft_model = AutoModelForCausalLM.from_pretrained(
        base_draft_model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    draft_model = PeftModel.from_pretrained(base_draft_model, draft_model_path)

    # Load the target model (large 7B model)
    target_model_path = "/home/asperger/DPSD/models/Qwen2.5-7B-Instruct"
    target_tokenizer = AutoTokenizer.from_pretrained(target_model_path, trust_remote_code=True)
    target_model = AutoModelForCausalLM.from_pretrained(
        target_model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    return draft_model, draft_tokenizer, target_model, target_tokenizer

def speculative_decoding(
    prompt, 
    draft_model, 
    draft_tokenizer, 
    target_model, 
    target_tokenizer, 
    max_length=256, 
    gamma=8, 
    temp=1.0
):
    """
    Speculative Decoding:
    - `gamma`: Number of draft tokens to generate per step.
    - `prompt`: Initial input text.
    - `draft_model` and `target_model`: The draft and target models for decoding.
    """
    # Initialization
    draft_device = draft_model.device
    target_device = target_model.device
    prefix = draft_tokenizer(prompt, return_tensors="pt").input_ids.to(draft_device)
    max_tokens = max_length
    total_draft_tokens = 0
    accepted_draft_tokens = 0
    
    while prefix.shape[1] < max_tokens:
        prefix_len = prefix.shape[1]
        
        # Step 1: Draft model generates `gamma` tokens
        draft_outputs = draft_model.generate(
            prefix, 
            max_new_tokens=gamma,
            do_sample=False
        )
        # Extract only the newly generated draft tokens
        draft_tokens = draft_outputs[:, prefix_len:]
        
        # Step 2: Target model evaluates probabilities for the same sequence
        extended_inputs = torch.cat([prefix, draft_tokens], dim=1).to(target_device)
        target_outputs = target_model(extended_inputs)
        target_logits = target_outputs.logits[:, prefix_len-1:, :]  # Only new tokens
        target_tokens = torch.argmax(target_logits, dim=-1)
        
        # Step 3: Verification process
        n = gamma
        for i in range(gamma):
            # Get the draft token at position `i` and calculate probabilities
            draft_token = draft_tokens[:, i]  # Token proposed by draft model
            target_token = target_tokens[:, i]

            if draft_token != target_token:
                n = i+1
                break

        # Update statistics
        total_draft_tokens += gamma  # Increment total tokens generated
        accepted_draft_tokens += n  # Increment accepted tokens
        
        # Accept tokens up to position `n`
        prefix = torch.cat([prefix, target_tokens[:, :n]], dim=1)
        
        if prefix.shape[1] >= max_tokens:
            break
    
    draft_pass_rate = accepted_draft_tokens / total_draft_tokens * 100 if total_draft_tokens > 0 else 0
    print(f"passed rate: {draft_pass_rate}%")
    # Decode final result
    decoded_text = draft_tokenizer.decode(prefix.squeeze(0), skip_special_tokens=True)
    return decoded_text


def main():
    # Load models and tokenizers
    draft_model, draft_tokenizer, target_model, target_tokenizer = load_models()

    # Input prompt
    prompt = "James decides to run 3 sprints 3 times a week.  He runs 60 meters each sprint.  How many total meters does he run a week?"

    # Perform speculative decoding
    generated_text = speculative_decoding(
        prompt, draft_model, draft_tokenizer, target_model, target_tokenizer, max_length=256
    )

    # print("Generated Text:", generated_text)

if __name__ == "__main__":
    main()
