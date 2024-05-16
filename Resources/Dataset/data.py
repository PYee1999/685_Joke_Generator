<<<<<<< HEAD
=======
from getPrompt import getPrompt
import pandas as pd
import numpy as np


def load(start, end):
    df_total = pd.read_csv('shortjokes.csv')
    df = df_total.iloc[start:end].copy()
    jokes = df['Joke'].values
    jokes = jokes.flatten()
    prompts = []
    api_key = '3f06bb7945192b6e58e9d4acd1d265fe51a1478015fb5cac5d67c3ca274b3c94'
    endpoint = 'https://api.together.xyz/v1/chat/completions'
    for joke in jokes:
        prompt = getPrompt(joke, api_key, endpoint)
        print(prompt, joke)
        prompts.append(prompt)

    df['Prompt'] = np.array(prompts)
    df.to_csv('testJokes.csv', mode='a', header=False, index=False)


for Start in range(150000, 150100, 100):
    End = Start + 100
    print(Start, End)
    load(Start, End)
>>>>>>> 2f4ac9e (add ratings.csv)
