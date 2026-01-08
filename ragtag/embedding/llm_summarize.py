from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from pathlib import Path

prompt = ["Explain vLLM like I'm a five-year-old"]

sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
# model = "casperhansen/llama-3-8b-instruct-awq"
model = "meta-llama/Llama-3.1-8B"

tokenizer = AutoTokenizer.from_pretrained(model)


def print_outputs(outputs):
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    print("-" * 80)


messages = [
    {
        "role": "system",
        "content": "You are a patient teacher who explains technical ideas to children.",
    },
    {"role": "user", "content": "Explain vLLM like I'm a five-year-old."},
]

tokenizer = AutoTokenizer.from_pretrained(model)
tokenizer.chat_template = Path(
    "ragtag/embedding/chat_templates/tool_chat_template_llama3.1_json.jinja"
).read_text()

print(tokenizer.chat_template)

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

llm = LLM(model=model, dtype="half", max_model_len=2048, swap_space=16)

outputs = llm.chat(prompt, sampling_params)

print_outputs(outputs)
