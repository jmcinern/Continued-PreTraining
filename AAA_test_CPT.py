from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
# comparing text generation in en and ga between base qwen and trained qwen
# base model
model_size = "0.6"
cache_path = "./cache/qwen3-"+model_size+"B"
model_name = "Qwen/Qwen3-"+model_size+"B"

# OG
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir=cache_path,
    trust_remote_code=True, 
    torch_dtype=torch.float16
)

# MINE
trained_model_path = "./checkpoints/checkpoint-139"
trained_model = AutoModelForCausalLM.from_pretrained(
    trained_model_path,
    trust_remote_code=True,
    local_files_only=True
    )

prompt_ga = "Is mise Seosamh, is feirmeoir mé, mar sin"
prompt_en = "My name is Joseph, I am a farmer, so"


tokenizer = AutoTokenizer.from_pretrained(
    f"Qwen/Qwen3-{model_size}B",
    trust_remote_code=True  
)

def generate_text(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=200,
            do_sample=True,
            top_p=0.9,
            temperature=0.8,
            repetition_penalty=1.2,
            top_k=50   
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)



print("Base Model Generated Text EN:")
generated_text = generate_text(base_model, tokenizer, prompt_en)
print(generated_text)

print("\n" + "="*50 + "\n")

print("Trained Model Generated Text EN:")
generated_text_trained = generate_text(trained_model, tokenizer, prompt_en)
print(generated_text_trained)

print("\n" + "="*50 + "\n")

print("Base Model Generated Text GA:")    
generated_text_ga = generate_text(base_model, tokenizer, prompt_ga)
print(generated_text_ga)

print("\n" + "="*50 + "\n")

print("Trained Model Generated Text GA:")    
generated_text_trained_ga = generate_text(trained_model, tokenizer, prompt_ga)
print(generated_text_trained_ga)

