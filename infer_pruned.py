import torch

from prune_llama import build_prompt, load_and_prune_model, normalize_text, read_text


recipe_path = "pruning_recipe.json"
system_prompt_path = "system_prompt_deep_learning.txt"
question = "What is the difference between overfitting and underfitting in deep learning?"
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
max_new_tokens = 128


model, tokenizer = load_and_prune_model(recipe_path, device, dtype)
system_prompt = read_text(system_prompt_path)
prompt = build_prompt(tokenizer, system_prompt, question)

inputs = tokenizer(prompt, return_tensors="pt").to(device)
gen_ids = model.generate(
    **inputs,
    max_new_tokens=max_new_tokens,
    do_sample=False,
    top_p=1.0,
    num_beams=1,
    pad_token_id=tokenizer.eos_token_id,
    use_cache=False,
)
prompt_len = inputs["input_ids"].shape[1]
gen_tokens = gen_ids[:, prompt_len:]
decoded = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
answer = normalize_text(decoded[0])
print(answer)
