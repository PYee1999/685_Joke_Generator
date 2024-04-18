from getPrompt import getPrompt
import pandas as pd
import numpy as np

df = pd.read_csv('shortjokes.csv')
start = 1010  # Replace with index of last generated datapoint. Check with jokes.csv to find last index.
end = start + 500  # Generates for 500 datapoints in one run (takes around 10 minutes).
df = df.iloc[start:end].copy()
jokes = df['Joke'].values
jokes = jokes.flatten()
prompts = []
api_key = ''
endpoint = ''
for joke in jokes:
    prompt = getPrompt(joke, api_key, endpoint)
    print(prompt, joke)
    prompts.append(prompt)

df['Prompt'] = np.array(prompts)
df.to_csv('jokes.csv', mode='a', header=False, index=False)
