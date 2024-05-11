from getRating import getRating
import pandas as pd
import numpy as np


def load(start, end):
    df_total = pd.read_csv('jokes.csv')
    df = df_total.iloc[start:end].copy()
    prompts = df['Prompt'].values
    jokes = df['Joke'].values
    jokes = jokes.flatten()
    ratings = []
    api_key = '3f06bb7945192b6e58e9d4acd1d265fe51a1478015fb5cac5d67c3ca274b3c94'
    endpoint = 'https://api.together.xyz/v1/chat/completions'
    for i in range(len(jokes)):
        joke = jokes[i]
        prompt = prompts[i]
        rating = getRating(joke, prompt, api_key, endpoint)
        print(prompt, joke, rating)
        ratings.append(rating)

    df['Rating'] = np.array(rating)
    df.to_csv('jokes.csv', mode='a', header=False, index=False)


for Start in range(6120, 6121, 100):
    End = Start + 1
    print(Start, End)
    load(Start, End)
