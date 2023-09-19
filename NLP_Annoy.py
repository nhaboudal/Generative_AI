import pandas as pd
import numpy as np
import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sentence_transformers import SentenceTransformer
from annoy import AnnoyIndex

# Load the dataset
url = "https://raw.githubusercontent.com/nhaboudal/NLP_FAQ/main/faqHP.csv"
df = pd.read_csv(url)
# Basic preprocessing
df['question'] = df['question'].str.lower()
df['answer'] = df['answer'].str.lower()

# Load pre-trained model and tokenizer for GPT-2
model_name = "gpt2-medium"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

def get_answer(question):
    input_ids = tokenizer.encode(question, return_tensors="pt")
    output = model.generate(input_ids, max_length=150, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    answer = tokenizer.decode(output[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    return answer

# Initialize the sentence transformer model
sentence_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def convert_to_vector(text):
    return sentence_model.encode([text])[0]

# Convert questions to vectors and build the Annoy index
dimension = 384  # Dimension of the vectors
index = AnnoyIndex(dimension, 'angular')

for i, question in enumerate(df['question'].tolist()):
    index.add_item(i, convert_to_vector(question))

index.build(10)  # 10 trees for the index. More trees give higher precision.

def search_question(query):
    query_vector = convert_to_vector(query)
    nearest_item_index = index.get_nns_by_vector(query_vector, 1)[0]
    return df.iloc[nearest_item_index]['answer']

# Streamlit UI
st.title("NLP-based FAQ Assistant")
user_input = st.text_input("Ask a question:")

if user_input:
    response = search_question(user_input)
    if not response:
        response = "Sorry, I couldn't find an answer."
    st.write(response)
