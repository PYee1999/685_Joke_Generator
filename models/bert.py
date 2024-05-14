# LINK: https://www.geeksforgeeks.org/explanation-of-bert-model-nlp/
# https://www.scaler.com/topics/nlp/bert-question-answering/

from dataset.process_data import preprocess_jokes
from models.train_data import convert_data_to_bert_input, normal_train
from transformers import BertTokenizer, BertForSequenceClassification, BertForQuestionAnswering # type: ignore
import torch # type: ignore

# Load the BERT model and tokenizer
# tokenizer = BertTokenizer.from_pretrained('deepset/bert-base-cased-squad2')
model = BertForQuestionAnswering.from_pretrained('deepset/bert-base-cased-squad2')


""" DATA PREPARATION """

# Get Jokes dataset
data = preprocess_jokes("./Dataset/jokes.csv")

# data = [("Hello, how can I help you today?", "Hi, I need some help with my order."),
#         ("Sure, what seems to be the problem?", "I never received my order and it's been over a week."),
#         ("I'm sorry to hear that. Let me check on that for you.", "Thank you. Can you also check on the status of my refund?"),
#         ("Certainly. I will check on that as well.", "Thank you. Can you also provide me with the contact information for your supervisor?"),
#         ("Of course. Here is the phone number and email address for our supervisor.", "Thank you for your help.")]

questions = [item[0] for item in data]
answers = [item[1] for item in data]

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

# Convert the data to input format for BERT
input_ids, attention_masks = convert_data_to_bert_input(device, tokenizer, questions, answers)


""" TRAINING """

# Set the labels for the data
labels = [0 if i < len(questions) else 1 for i in range(len(questions) + len(answers))]
labels = torch.tensor(labels).to(device)

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
