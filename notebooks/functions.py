# Imports
import os
import re
import pandas as pd
import numpy as np

import torch
import gc
import pickle
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, Trainer, TrainingArguments
import bitsandbytes as bnb
from datasets import Dataset

# ========== Functions ========== #
def generate_file_structure(start_path, indent=''):
  """
  Generate a file structure tree for a given directory path... to copy&paste on the README.md file...
  """
  file_structure = ''
  for item in os.listdir(start_path):
      if item == '.git':
        continue # Skip the .git folder
      
      item_path = os.path.join(start_path, item)
      if os.path.isdir(item_path):
        file_structure += f'{indent}├── {item}\n'
        file_structure += generate_file_structure(item_path, indent + '│   ')
      else:
        file_structure += f'{indent}├── {item}\n'
  return file_structure


# ========== Cleaning Functions ========== #
# Define punctuation to keep
keep_punctuation = {".", ",", "!", "?", "'"}

# Cleaning function
def clean_text(review):
  review = review.lower()  # Lowercase

  # Remove unwanted characters (keep only letters, numbers, and whitelisted punctuation)
  cleaned_text = "".join(char if char.isalnum() or char in keep_punctuation or char.isspace() else " " for char in review)

  # Remove extra spaces (from removed characters)
  cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

  return cleaned_text


# ========== Prompt Functions ========== #
def create_prompt(text, emotions, label):
  """
  Create a prompt for the model to generate a response to a customer review.
  """
  text =  f"A customer left us a {label}-star review: '{text}' The customer feels {emotions}. Concisely, how can we best improve our services for this customer's experience?"
  return f"[INST] {text} [/INST]"

def clean_response(text): 
  """
  Clean the response text by removing the special tokens and extra spaces.
  """
  # Remove any occurrences of '</s>' (special token for end of sequence)
  text = text.replace("</s>", "").strip()

  # Find last period
  last_period = text.rfind('.')
  if last_period == -1:
      return text

  # Get text up to last period
  cleaned = text[:last_period + 1]

  # Check for numbered list pattern at the end (like "5." with nothing after it)
  pattern = r'\n\d+\.\s*$'
  cleaned = re.sub(pattern, '', cleaned)

  return cleaned.strip()

def clean_output(input_text, output_text):
  """
  Cleans the output text by removing the input text from it.
  """
  # Normalize whitespace
  input_norm = re.sub(r'\s+', ' ', input_text.strip())
  output_norm = re.sub(r'\s+', ' ', output_text.strip())

  # Escape special regex characters in input text
  pattern = re.escape(input_norm)

  # Remove input from output using regex (case-insensitive)
  cleaned_output = re.sub(pattern, '', output_norm, flags=re.IGNORECASE).strip()

  # Final cleanup: remove extra spaces
  return re.sub(r'\s+', ' ', cleaned_output).strip()

def clean_phi2_output(input_text, output_text):
    """
    Extracts the response from the model output by removing the input prompt.
    Works by detecting markers like "Answer:", "Response:", "###", or newline separation.
    """
    # Normalize whitespace
    input_norm = re.sub(r'\s+', ' ', input_text.strip())
    output_norm = re.sub(r'\s+', ' ', output_text.strip())

    # Look for common response markers
    response_markers = [r"Answer:", 
                        r"Response:", 
                        r"###", 
                        r"\n\n", 
                        r"##Your task:",
                        r"<|Question|>",
                        r"<|Answer|>",
                        r"Instruction: ",
                        r"Response: ",
                        r"OUTPUT:"]
    
    # Try to find the first occurrence of any marker
    for marker in response_markers:
        match = re.search(marker, output_norm, re.IGNORECASE)
        if match:
            return output_norm[match.end():].strip()  # Extract everything after the marker

    # If no marker is found, attempt a basic subtraction
    cleaned_output = re.sub(re.escape(input_norm), '', output_norm, flags=re.IGNORECASE).strip()
    
    return cleaned_output if cleaned_output else output_norm  # Default to original if no cleaning was effective

# ========== LLM-related Functions ========== #

def get_emotion_label(text):
  """
  Get the top 3 predicted emotions for a given text.
  """
  # Tokenize text
  tokens = tokenizer(text,
                       padding=True,
                       truncation=True,
                       max_length=128,
                       return_tensors="pt").to(model.device)

  # Run inference
  with torch.no_grad():
        outputs = model(**tokens)

  # Get top 3 predicted labels
  probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)  # Convert logits to probabilities
  top3_indices = torch.argsort(probabilities, descending=True)[0][:3]  # Get top 3 indices
    
  # Convert indices to emotion labels
  top3_emotions = [emotion_labels[i] for i in top3_indices]
    
  # Return as string...
  return ", ".join(top3_emotions)