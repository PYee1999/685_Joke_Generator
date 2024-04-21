from getPrompt import getPrompt
import pandas as pd # type: ignore
import numpy as np # type: ignore

df = pd.read_csv('shortjokes.csv')
start = 1012  # Replace with index of last generated datapoint. Check with jokes.csv to find last index.
end = start + 10  # Generates for 500 datapoints in one run (takes around 10 minutes).
df = df.iloc[start:end].copy()
jokes = df['Joke'].values
jokes = jokes.flatten()
prompts = []
api_key = '3f06bb7945192b6e58e9d4acd1d265fe51a1478015fb5cac5d67c3ca274b3c94'
endpoint = 'https://api.together.xyz/v1/chat/completions'
model = 'meta-llama/Llama-3-70b-chat-hf'
for joke in jokes:
    prompt = getPrompt(joke, api_key, endpoint, model)
    print("Prompt: ", prompt)
    print("Joke: ", joke)
    prompts.append(prompt)

df['Prompt'] = np.array(prompts)
df.to_csv('jokes.csv', mode='a', header=False, index=False)
