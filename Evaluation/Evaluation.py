from  getEvaluation import getEval
import pandas as pd
import numpy as np


def loadRating(start, end):
    df_total = pd.read_csv('../Evaluation/jokes.csv')
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
        print(prompt, joke, rating)
        ratings.append(rating)

    df['Rating'] = np.array(rating)
    df.to_csv('ratings.csv', mode='a', header=False, index=False)


def loadEval(start, end):
    df_total = pd.read_csv('GeneratedJokes.csv')
    df = df_total.iloc[start:end].copy()
    prompts = df.iloc[:, 1].values.flatten()
    pretrained_jokes = df.iloc[:, 2].values.flatten()
    finetuned_jokes = df.iloc[:, 3].values.flatten()
    best_jokes = []
    api_key = '3f06bb7945192b6e58e9d4acd1d265fe51a1478015fb5cac5d67c3ca274b3c94'
    endpoint = 'https://api.together.xyz/v1/chat/completions'
    for i in range(len(prompts)):
        pretrained_joke = pretrained_jokes[i]
        finetuned_joke = finetuned_jokes[i]
        prompt = prompts[i]
        joke = getEval(prompt, pretrained_joke, finetuned_joke, api_key, endpoint)
        print(prompt, joke)
        best_jokes.append(joke)

    df['BestJoke'] = np.array(best_jokes)
    df.to_csv('evaluation.csv', mode='a', header=False, index=False)


for Start in range(1, 5):
    End = Start + 1
    print(Start, End)
    loadEval(Start, End)
