from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载模型和分词器
model_name = "meta-llama/Llama-2-7b-chat-hf"  # 或其他模型
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    mirror="https://hf-mirror.com"
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    mirror="https://hf-mirror.com",
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True
)

# 生成文本
prompt = "请解释量子力学的基本原理"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(
    inputs.input_ids,
    max_new_tokens=512,
    do_sample=True,
    temperature=0.7
)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)