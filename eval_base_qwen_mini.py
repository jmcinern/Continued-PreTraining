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
        max_length=500,
        do_sample=True,
        top_k=50,
        top_p=0.9,
        temperature=0.7,
        no_repeat_ngram_size=3,
    )

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Save to file
with open("base_irish_politics_story.txt", "w", encoding="utf-8") as f:
    f.write(generated_text)

model_tested = "qwen3-0.6B-CPT_ga_ALL_DATA_7e"
model_path = "./checkpoints/" + model_tested
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
model.eval()

prompt = '''Bhí trí chéad fear ag baint cloiche i nGleann na Smól. Ní raibh ag éirí leo. Thug siad faoi deara 
fear mór dathúil ar mhuin chapaill bháin. Thug sé cabhair do na fir. Chrom sé síos agus rug greim 
ar an gcarraig mhór. Chaith sé uaidh í gan stró ar bith. Bhris giorta an chapaill, áfach agus thit 
sé go talamh. Rith an chapall leis ar chosa in airde agus fágadh an fear ina luí, ina sheanfhear 
críonna, caite dall. Tugadh chuig Naomh Pádraig an seanduine. D’inis sé le Naomh Pádraig gurb 
é Oisín i ndiaidh na Féinne é. D’inis sé a scéal dó.'''

inputs = tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_length=500,
        do_sample=True,
        top_k=50,
        top_p=0.9,
        temperature=0.7,
        no_repeat_ngram_size=3,
    )

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Save to file
with open("test_texts/cpt_oisin_achoimre_"+model_tested+".txt", "w", encoding="utf-8") as f:
    f.write(generated_text)

print(generated_text)