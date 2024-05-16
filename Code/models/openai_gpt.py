import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

jokes_dataset = pd.read_csv("../Dataset/jokes.csv", on_bad_lines='skip', header=None)
jokes_dataset = jokes_dataset.sample(frac=1).reset_index(drop=True)
jokes_dataset = jokes_dataset.iloc[:,1:3].values

preprocessed_data = []
for joke, prompt in jokes_dataset[:100]:
    preprocessed_data.append(f"User: {prompt}\n")
    preprocessed_data.append(f"Assistant: {joke}\n")
preprocessed_data = "".join(preprocessed_data)

output_file = "../Dataset/jokes.txt"
with open(output_file, "w") as f:
    f.write(preprocessed_data)

model_name = "gpt2"
train_file = "../Dataset/jokes.txt"
output_dir = "output"

# Load GPT-2 model and tokenizer
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

from transformers import pipeline, set_seed
generator = pipeline('text-generation', model='gpt2')
set_seed(42)
print("Output before fine-tuning:", generator("Tell me a joke about cats,", max_length=30, num_return_sequences=5))

# Load training dataset
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