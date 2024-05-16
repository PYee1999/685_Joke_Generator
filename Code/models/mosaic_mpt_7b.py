# LINK: https://huggingface.co/mosaicml/mpt-7b

from dataset.process_data import create_joke_file, preprocess_jokes # type: ignore
from models.train_data import convert_data_to_bert_input, normal_train # type: ignore
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, pipeline, set_seed, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments # type: ignore
import torch # type: ignore

# Name of model
name = 'mosaicml/mpt-7b'

config = AutoConfig.from_pretrained(name, trust_remote_code=True)
config.max_seq_len = 4096 # (input + output) tokens can now be up to 4096

# Create model
model = AutoModelForCausalLM.from_pretrained(
  name,
  config=config,
  trust_remote_code=True
)

# Create tokenizer
tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b')

generator = pipeline('text-generation', model='gpt2')
set_seed(42)
print("Output before fine-tuning:", generator("Tell me a joke about cats,", max_length=30, num_return_sequences=5))

# Load training dataset
# train_file, output_dir = create_joke_file()
train_file = "./dataset/jokes.txt"
output_dir = "output"

train_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path=train_file,
    block_size=128)

# Create data collator for language modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False)

# Set training arguments
training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=5,
    save_steps=60,
    save_total_limit=2,
)

# Train the model
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
)

trainer.train()

# Save the fine-tuned model
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print("Output before fine-tuning:", generator("Tell me a joke about cats,", max_length=30, num_return_sequences=5))

quit()

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get Jokes dataset
data = preprocess_jokes("./Dataset/jokes.csv")
questions = [item[0] for item in data]
answers = [item[1] for item in data]

# Convert the data to input format for BERT
input_ids, attention_masks = convert_data_to_bert_input(device, tokenizer, questions, answers)

# Set the labels for the data
labels = [0 if i < len(questions) else 1 for i in range(len(questions) + len(answers))]
labels = torch.tensor(labels).to(device)

""" TRAINING """

normal_train(input=input_ids, 
             labels=labels, 
             model=model, 
             optim="adam", 
             learning_rate=1e-4, 
             criterion=torch.nn.CrossEntropyLoss(), 
             num_epochs=10, 
             attention_masks=attention_masks)


""" MAKE PREDICTION """

max_length = 512

# Set the model to eval mode
model.eval()

# Define the input
input_text = "Tell me a joke about dogs"
print("Input:", input_text)

# Tokenize the input
input_tokens = tokenizer.tokenize(input_text)

# Pad the input tokens
input_tokens = input_tokens + [tokenizer.pad_token] * (max_length - len(input_tokens))

# Convert the input tokens to input ids
input_ids = tokenizer.convert_tokens_to_ids(input_tokens)

# Create the attention mask for the input
attention_mask = [1 if token != tokenizer.pad_token else 0 for token in input_tokens]

# Convert the input ids and attention mask to tensors
input_ids = torch.tensor(input_ids).unsqueeze(0).to(device)
attention_mask = torch.tensor(attention_mask).unsqueeze(0).to(device)

# Get the model output
output = model(input_ids, attention_mask=attention_mask)

# Get the predicted label
prediction = output[0].argmax(dim=1).item()

# Print the output
if prediction == 0:
    print("Question: {}".format(input_text))
else:
    print("Answer: {}".format(answers[prediction - 1]))
