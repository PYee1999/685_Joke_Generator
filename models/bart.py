# BART Source
# https://huggingface.co/facebook/bart-large
# https://huggingface.co/docs/transformers/en/model_doc/bart
# https://huggingface.co/transformers/v3.0.2/model_doc/bart.html

# Question Answering
# https://huggingface.co/aware-ai/bart-squadv2#:~:text=To%20use%20BART%20for%20question,comparable%20to%20ROBERTa%20on%20SQuAD.

from transformers import BartTokenizer, BartForQuestionAnswering
import torch

tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
model = BartForQuestionAnswering.from_pretrained('facebook/bart-large')

params = {
    "inputs": "Hello, my dog is cute",
    "parameters": {
        "repetition_penalty": 4.0,
        "max_length": 128
    }
}

inputs = tokenizer(params["inputs"], return_tensors="pt")
start_positions = torch.tensor([1])
end_positions = torch.tensor([3])

outputs = model(**inputs, start_positions=start_positions, end_positions=end_positions)
loss, start_scores, end_scores = outputs[:3]

input_ids = inputs['input_ids']

# Generate answer
all_tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
answer = ' '.join(all_tokens[torch.argmax(start_scores) : torch.argmax(end_scores)+1])
answer = tokenizer.convert_tokens_to_ids(answer.split())
answer = tokenizer.decode(answer)
print("Answer:", answer)
#answer => 'a nice puppet' 