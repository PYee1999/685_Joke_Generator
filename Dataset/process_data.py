import pandas as pd # type: ignore

def preprocess_jokes(dir: str) -> list:
    # Extract Jokes CSV dataset
    jokes_dataset = pd.read_csv(dir, header=None)
    jokes_dataset = jokes_dataset.sample(frac=1).reset_index(drop=True)
    jokes_dataset = jokes_dataset.iloc[:,1:3].values

    # Convert data into a list of tuples
    data = []
    for joke, prompt in jokes_dataset[:100]:
        # Tuple = (Prompt for LLM, Answer as Joke)
        pair = (prompt, joke)
        data.append(pair)

    # Return list
    return data


def export_jokes_txt(df):
    txt_filename = "./dataset/jokes.txt"

    jokes_dataset = df.sample(frac=1).reset_index(drop=True)
    jokes_dataset = jokes_dataset.iloc[:,1:3].values

    preprocessed_data = []
    for joke, prompt in jokes_dataset[:100]:
        preprocessed_data.append(f"User: {prompt}\n")
        preprocessed_data.append(f"Assistant: {joke}\n")
    preprocessed_data = "".join(preprocessed_data)
    print(preprocessed_data)

    with open(txt_filename, "w") as f:
        f.write(preprocessed_data)
    print("txt file exported")

    train_file = "./dataset/jokes.txt"
    output_dir = "output"

    return train_file, output_dir


def create_joke_file():
    csv_file = "dataset/jokes.csv"
    
    jokes_dataset = pd.read_csv(csv_file, on_bad_lines='skip', header=None)
    jokes_dataset = jokes_dataset.sample(frac=1).reset_index(drop=True)
    jokes_dataset = jokes_dataset.iloc[:,1:3].values
    # print(jokes_dataset)

    preprocessed_data = []
    for joke, prompt in jokes_dataset[:100]:
        preprocessed_data.append(f"User: {prompt}\n")
        preprocessed_data.append(f"Assistant: {joke}\n")
    preprocessed_data = "".join(preprocessed_data)
    print(preprocessed_data)

    with open(csv_file, "w") as f:
        f.write(preprocessed_data)
    print("txt file exported")

    train_file = "./dataset/jokes.txt"
    output_dir = "output"

    return train_file, output_dir