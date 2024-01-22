#! /bin/env python3


# from transformers import TextGenerationModel, AutoTokenizer
from transformers import LlamaForCausalLM, LlamaTokenizer
import os

# Define your model and tokenizer names
# model_name = "text-bison@001"
# tokenizer_name = "text-bison@001"
model = LlamaForCausalLM.from_pretrained("/output/path")
tokenizer = LlamaTokenizer.from_pretrained("/output/path")


# Specify a directory to save the model locally
local_model_dir = "local_model"

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

# Create a directory to save the model if it doesn't exist
os.makedirs(local_model_dir, exist_ok=True)

# Check if the model is already saved locally, and save it if not
if not os.path.exists(local_model_dir):
    # Download and save the model locally
    model = TextGenerationModel.from_pretrained(model_name)
    model.save_pretrained(local_model_dir)

# Load the model from the local directory
model = TextGenerationModel.from_pretrained(local_model_dir)

# Define a prompt for text generation
prompt = "What's the capital of Indonesia?"
encoded_prompt = tokenizer.encode(prompt, return_tensors="pt")

# Generate text locally
generated_text = model.generate(encoded_prompt, max_length=50, num_return_sequences=1)

# Print the generated text
print(generated_text[0]["text"])



"""
import pandas as pd
from vertexai.language_models import TextGenerationModel

generation_model = TextGenerationModel.from_pretrained("text-bison@001")

prompt = "What's the capital of Indonesia?"
print(
    generation_model.predict(
        prompt,
        max_output_tokens=256,
        temperature=0.1,
    ).text
)

"""
