# LINK: https://www.geeksforgeeks.org/explanation-of-bert-model-nlp/
# https://www.scaler.com/topics/nlp/bert-question-answering/

from transformers import BertTokenizer, BertForSequenceClassification, BertForQuestionAnswering
import torch

# Load the BERT model and tokenizer
# tokenizer = BertTokenizer.from_pretrained('deepset/bert-base-cased-squad2')
model = BertForQuestionAnswering.from_pretrained('deepset/bert-base-cased-squad2')


""" DATA PREPARATION """

data = [("Hello, how can I help you today?", "Hi, I need some help with my order."),
        ("Sure, what seems to be the problem?", "I never received my order and it's been over a week."),
        ("I'm sorry to hear that. Let me check on that for you.", "Thank you. Can you also check on the status of my refund?"),
        ("Certainly. I will check on that as well.", "Thank you. Can you also provide me with the contact information for your supervisor?"),
        ("Of course. Here is the phone number and email address for our supervisor.", "Thank you for your help.")]

questions = [item[0] for item in data]
answers = [item[1] for item in data]

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

# Convert the data to input format for BERT
question_input_ids = []
question_attention_masks = []
for i in range(len(questions)):
    # Tokenize the question
    question_tokens = tokenizer.tokenize(questions[i])
    
    max_length = 512
    # Pad the input tokens
    question_tokens = question_tokens + [tokenizer.pad_token] * (max_length - len(question_tokens))

    # Create the input ids for the BERT model
    question_input_ids.append(tokenizer.convert_tokens_to_ids(question_tokens))

    # Create the attention masks for the input tokens
    question_attention_masks.append([1 if token != tokenizer.pad_token else 0 for token in question_tokens])

answer_input_ids = []
answer_attention_masks = []
for i in range(len(answers)):
    # Tokenize the answer
    answer_tokens = tokenizer.tokenize(answers[i])

    # Pad the input tokens
    answer_tokens = answer_tokens + [tokenizer.pad_token] * (max_length - len(answer_tokens))

    # Create the input ids for the BERT model
    answer_input_ids.append(tokenizer.convert_tokens_to_ids(answer_tokens))

    # Create the attention masks for the input tokens
    answer_attention_masks.append([1 if token != tokenizer.pad_token else 0 for token in answer_tokens])

# Concatenate the question and answer input lists
input_ids = question_input_ids + answer_input_ids
attention_masks = question_attention_masks + answer_attention_masks

# Convert the input ids and attention masks to tensors
input_ids = torch.tensor(input_ids).to(device)
attention_masks = torch.tensor(attention_masks).to(device)


""" TRAINING """

# Define the criterion
criterion = torch.nn.CrossEntropyLoss()

# Define optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

def calculate_accuracy(predictions, labels):
    correct_predictions = 0
    for i in range(len(predictions)):
        if predictions[i] == labels[i]:
            correct_predictions += 1
    return correct_predictions / len(predictions)

# Set the number of epochs
num_epochs = 5

# Set the labels for the data
labels = [0 if i < len(questions) else 1 for i in range(len(questions) + len(answers))]
labels = torch.tensor(labels).to(device)

# Set the training loop
for epoch in range(num_epochs):

    # Set the training mode
    model.train()

    # Clear the gradients
    model.zero_grad()

    # Forward pass
    output = model(input_ids, attention_mask=attention_masks)

    # Calculate the loss
    loss = criterion(output[0], labels)

    # Backward pass
    loss.backward()

    # Update the parameters
    optimizer.step()

    # Print the loss and accuracy
    print("Epoch {}/{} - Loss: {:.5f} - Accuracy: {:.5f}".format(epoch + 1, num_epochs, loss.item(), calculate_accuracy(output[0].argmax(dim=1).cpu().numpy(), labels.cpu().numpy())))


""" MAKE PREDICTION """

# Set the model to eval mode
model.eval()

# Define the input
input_text = "I'm sorry to hear that. Let me check on that for you."

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
