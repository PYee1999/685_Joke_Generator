from getPrompt import getPrompt
import pandas as pd
import numpy as np


def load(new_df, file_name):
    jokes = new_df['Joke'].values
    jokes = jokes.flatten()
    jokes = np.array([str(x).encode('ascii', 'ignore').decode() for x in jokes])
    prompts = []
    new_jokes = []
    api_key = '3f06bb7945192b6e58e9d4acd1d265fe51a1478015fb5cac5d67c3ca274b3c94'
    endpoint = 'https://api.together.xyz/v1/chat/completions'
    for joke in jokes:
        prompt = getPrompt(joke, api_key, endpoint)
        if prompt is None:
            continue
        prompt = prompt.split('.')[0]
        print(prompt, joke)
        prompts.append(prompt)
        new_jokes.append(joke)
    newer_df = pd.DataFrame()
    newer_df['Jokes'] = np.array(new_jokes)
    newer_df['Prompts'] = np.array(prompts)
    newer_df.to_csv('../../Resources/Dataset/' + file_name, mode='a', header=False, index=False)


df_total = pd.read_csv('../../Resources/shortjokes.csv')
for start in range(5900, 10000, 100):
    end = start + 100
    df = df_total.iloc[start:end].copy()
    df.drop(columns=df.columns[0], inplace=True)
    load(df, file_name='jokes.csv')
    print(start)
