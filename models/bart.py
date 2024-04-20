# BART Source
# https://huggingface.co/facebook/bart-large
# https://huggingface.co/docs/transformers/en/model_doc/bart

# Question Answering
# https://huggingface.co/aware-ai/bart-squadv2#:~:text=To%20use%20BART%20for%20question,comparable%20to%20ROBERTa%20on%20SQuAD.

from transformers import BartTokenizer, BartForQuestionAnswering
import torch

# Training arguments (For fine-tuning)
train_args = {
    'learning_rate': 1e-5,
    'max_seq_length': 512,
    'doc_stride': 512,
    'overwrite_output_dir': True,
    'reprocess_input_data': False,
    'train_batch_size': 8,
    'num_train_epochs': 2,
    'gradient_accumulation_steps': 2,
    'no_cache': True,
    'use_cached_eval_features': False,
    'save_model_every_epoch': False,
    'output_dir': "bart-squadv2",
    'eval_batch_size': 32,
    'fp16_opt_level': 'O2',
}

# Get model and tokenizer
tokenizer = BartTokenizer.from_pretrained('a-ware/bart-squadv2')
model = BartForQuestionAnswering.from_pretrained('a-ware/bart-squadv2')

# Set up question and answer, follow by encoding it
question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
encoding = tokenizer(question, text, return_tensors='pt')
print(f"encoding:\n{encoding}")
input_ids = encoding['input_ids']
attention_mask = encoding['attention_mask']

# Run input on model
start_scores, end_scores = model(input_ids, attention_mask=attention_mask, output_attentions=False)[:2]

# Get all tokoens
all_tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

# Generate answer
answer = ' '.join(all_tokens[torch.argmax(start_scores) : torch.argmax(end_scores)+1])
answer = tokenizer.convert_tokens_to_ids(answer.split())
answer = tokenizer.decode(answer)
print("answer:", answer)
#answer => 'a nice puppet' 

