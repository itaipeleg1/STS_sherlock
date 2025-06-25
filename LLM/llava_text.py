import os
import torch
import pandas as pd
from tqdm import tqdm
from transformers import pipeline

# Set Hugging Face cache dirs
os.environ['HF_HOME'] = '/home/new_storage/sherlock/hf_cache'
os.environ['TRANSFORMERS_CACHE'] = '/home/new_storage/sherlock/hf_cache'
os.environ['HUGGINGFACE_HUB_CACHE'] = '/home/new_storage/sherlock/hf_cache'

# Load Zephyr model as chat pipeline
pipe = pipeline("text-generation", model="HuggingFaceH4/zephyr-7b-alpha", torch_dtype=torch.float16, device_map="auto")

# Generate yes/no label from text
def get_yes_label_from_response(text):
    messages = [
        {
            "role": "system",
            "content": "You are restricted to answering with one word: 'yes' or 'no'.",
        },
        {
            "role": "user",
            "content": f"This text is describing a frame from a movie. Does this text suggests social interaction?\n\n{text}"
        },
    ]
    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    outputs = pipe(prompt, max_new_tokens=1, do_sample=False, temperature=0)

    full_text = outputs[0]["generated_text"]

    # Extract everything after <|assistant|>
    if "<|assistant|>" in full_text:
        response = full_text.split("<|assistant|>")[-1].strip().lower()
    return response

# Entry point
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute binary labels for social interaction.")
    parser.add_argument("input_file", type=str, help="Path to the input CSV file.")
    args = parser.parse_args()

    df = pd.read_csv(args.input_file)
    yes_labels = []
    raw_responses = []

    for text in tqdm(df['response'], desc="Classifying responses"):
        response = get_yes_label_from_response(text)
        print(f"Text: {text[:50]}... Response: {response}")
        if "yes" in response or "Yes" in response:
            label = 1
        else:
            label = 0
        yes_labels.append(label)


    df['yes_label'] = yes_labels


    output_file = f"{os.path.splitext(args.input_file)[0]}_with_labels.csv"
    df.to_csv(output_file, index=False)
    print(f"✅ Saved with 'yes_label' columns to: {output_file}")
