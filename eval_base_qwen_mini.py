from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "Qwen/Qwen3-0.6B"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
model.eval()

prompt = "Inis dom scéal faoi pholaitíocht na hÉireann."

inputs = tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_length=200,
        do_sample=True,
        top_p=0.9,
        temperature=0.8
    )

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Save to file
with open("base_irish_politics_story.txt", "w", encoding="utf-8") as f:
    f.write(generated_text)
