# LINK: https://www.geeksforgeeks.org/explanation-of-bert-model-nlp/

from transformers import BertTokenizer
 
# Load pre-trained BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
 
# Input text
text = 'ChatGPT is a language model developed by OpenAI, based on the GPT (Generative Pre-trained Transformer) architecture. '
 
# Tokenize and encode the text
encoding = tokenizer.encode(text)
 
# Print the token IDs
print("Token IDs:", encoding)
 
# Convert token IDs back to tokens
tokens = tokenizer.convert_ids_to_tokens(encoding)
 
# Print the corresponding tokens
print("Tokens:", tokens)