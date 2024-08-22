import argparse
import logging

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

parser = argparse.ArgumentParser(
    description="Perform inference with a Hugging Face model"
)
parser.add_argument(
    "--model-name-or-path",
    type=str,
    required=True,
    help="The name or path of the model to use",
)
parser.add_argument("--max-length", type=int, default=1024, help="The maximum length to generate")
parser.add_argument(
    "--temperature", type=float, default=1.0, help="The temperature for sampling"
)
parser.add_argument(
    "--top-p", type=float, default=1.0, help="The top-p value for nucleus sampling"
)
parser.add_argument(
    "--repetition-penalty", type=float, default=1.0, help="The repetition penalty"
)
args = parser.parse_args()

logging.basicConfig(
    level=logging.INFO,
    format="| %(levelname)s | %(asctime)s | %(name)s | %(message)s",
)

logging.info("Check for mixed precision support")
if torch.cuda.is_available():
    if torch.cuda.is_bf16_supported():
        torch_dtype = torch.bfloat16
    elif torch.cuda.is_fp16_supported():
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32
else:
    torch_dtype = torch.float32
logging.info(f"Selected dtype: {torch_dtype}")

logging.info("Load tokenizer")
tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

logging.debug("Load model")
model = AutoModelForCausalLM.from_pretrained(
    args.model_name_or_path, device_map="auto", torch_dtype=torch_dtype
)
model.eval()

logging.info("Start inference loop")
while True:
    prompt = input("Prompt >>> ")
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"]
    outputs = model.generate(
        input_ids,
        do_sample=True,
        max_length=args.max_length,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
    )
    output_ids = outputs[0][input_ids.size(1) :]
    generated_text = tokenizer.decode(output_ids)
    print(generated_text)
