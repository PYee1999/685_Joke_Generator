from tqdm import tqdm # type: ignore
import torch # type: ignore


def calculate_accuracy(predictions, labels):
    """
    Calculate Accuracy of Training for each epoch
    :param predictions: 
    :param labels: 
    Return: Calculated accuracy results
    """
    correct_predictions = 0
    for i in range(len(predictions)):
        if predictions[i] == labels[i]:
            correct_predictions += 1
    return correct_predictions / len(predictions)


def convert_data_to_bert_input(device, tokenizer, question_list, answer_list, max_length=512):
    """
    Convert data from preprocess to tensors for training
    :param tokenizer: tokenizer for LLM
    :param question_list: list of question/prompts
    :param answer_list: list of answers/jokes
    :param max_length: 
    Return: Tuple of input tensor and attention_mask tensor
    """
    
    def convert_data_to_input_ids(token_list):
        """
        Convert token_list to list of ids
        """
        input_ids = []
        attention_masks = []
        
        for i in range(len(token_list)):
            # Tokenize the question
            question_tokens = tokenizer.tokenize(token_list[i])
            
            # Pad the input tokens
            question_tokens = question_tokens + [tokenizer.pad_token] * (max_length - len(question_tokens))

            # Create the input ids for the BERT model
            input_ids.append(tokenizer.convert_tokens_to_ids(question_tokens))

            # Create the attention masks for the input tokens
            attention_masks.append([1 if token != tokenizer.pad_token else 0 for token in question_tokens])

        return input_ids, attention_masks

    # Generate question ids
    question_input_ids, question_attention_masks = convert_data_to_input_ids(token_list=question_list)

    # Generate answer ids
    answer_input_ids, answer_attention_masks = convert_data_to_input_ids(token_list=answer_list)

    # Concatenate the question and answer input lists
    combined_ids = question_input_ids + answer_input_ids
    combined_masks = question_attention_masks + answer_attention_masks

    # Convert the input ids and attention masks to tensors
    combined_ids = torch.tensor(combined_ids).to(device)
    combined_masks = torch.tensor(combined_masks).to(device)

    # Return ids and mask
    return combined_ids, combined_masks


def normal_train(input, labels, model, optim="adam", learning_rate=1e-4, criterion=torch.nn.CrossEntropyLoss(), num_epochs=5, attention_masks=None):
    """
    Normal Training/Fine-Tuning
    :param input: data to input into model
    :param labels: data to compare to generated model output with criterion
    :param model: Model to train
    :param optim: optimizer to run (Default: Adam)
    :param learning_rate: learning rate for optimizer
    :param criterion: loss function (Default: Cross Entropy Loss)
    :param num_epochs: Number of epochs to run
    :param attention_masks: 
    Return: 
    """
    
    # Create optimizer
    if optim.lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optim.lower() == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    else:
        raise NotImplementedError("Optimizer does not exist")

    # Set the training loop
    for epoch in tqdm(range(num_epochs)):

        # Clear the gradients
        model.zero_grad()

        # Set the training mode
        model.train()

        # Forward pass
        output = model(input, attention_mask=attention_masks)

        # Calculate the loss
        loss = criterion(output[0], labels)

        # Backward pass
        loss.backward()

        # Update the parameters
        optimizer.step()

        # Print the loss and accuracy
        print("Epoch {}/{} - Loss: {:.5f} - Accuracy: {:.5f}".format(epoch + 1, num_epochs, loss.item(), calculate_accuracy(output[0].argmax(dim=1).cpu().numpy(), labels.cpu().numpy())))
