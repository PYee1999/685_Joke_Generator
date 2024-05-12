from getEvaluation import getRating
import pandas as pd
import numpy as np


def loadRating(start, end):
    df_total = pd.read_csv('../Evaluation/jokes_test_dataset.csv')
    df = df_total.iloc[start:end].copy()
    prompts = df.iloc[:, 2].values
    jokes = df.iloc[:, 1].values
    jokes = jokes.flatten()
    prompts = prompts.flatten()
    ratings = []
    api_key = '3f06bb7945192b6e58e9d4acd1d265fe51a1478015fb5cac5d67c3ca274b3c94'
    endpoint = 'https://api.together.xyz/v1/chat/completions'
    for i in range(len(jokes)):
        joke = jokes[i]
        prompt = prompts[i]
        rating = getRating(joke, prompt, api_key, endpoint)
        print(rating)
        ratings.append(rating)

    df['Rating'] = np.array(ratings)
    df.to_csv('ratings.csv', mode='a', header=False, index=False)


loadRating(100, 101)
