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