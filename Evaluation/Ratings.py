from getMetrics import getRating
import pandas as pd
import numpy as np
import os


def loadRating(start, end, file_name, folder_path):
    df_total = pd.read_csv(folder_path + file_name + '.csv')
    df = df_total.iloc[start:end].copy()
    prompts = df['Prompts'].values
    jokes = df['Jokes'].values
    jokes = jokes.flatten()
    prompts = prompts.flatten()
    promptScores = []
    funnyScores = []
    grammarScores = []
    lengthScores = []
    totalScores = []
    api_key = '3f06bb7945192b6e58e9d4acd1d265fe51a1478015fb5cac5d67c3ca274b3c94'
    endpoint = 'https://api.together.xyz/v1/chat/completions'
    for i in range(len(jokes)):
        joke = jokes[i]
        prompt = prompts[i]
        [promptScore, funnyScore, grammarScore, lengthScore] = getRating(joke, prompt, api_key, endpoint)
        print(i+1, [promptScore, funnyScore, grammarScore, lengthScore])
        promptScores.append(promptScore)
        funnyScores.append(funnyScore)
        grammarScores.append(grammarScore)
        lengthScores.append(lengthScore)
        totalScores.append(promptScore + funnyScore + grammarScore + lengthScore)

    df['promptScores'] = np.array(promptScores)
    df['funnyScores'] = np.array(funnyScores)
    df['grammarScores'] = np.array(grammarScores)
    df['lengthScores'] = np.array(lengthScores)
    df['totalScores'] = np.array(totalScores)
    print("Rating Mean Score", np.mean(np.array(totalScores)) * 100/6)
    df.to_csv('../LLMEvaluation/' + file_name + '_llm_evaluation.csv', mode='a', header=False, index=False)


folder_path = '../LLMJokesData/'
files_in_folder = os.listdir(folder_path)
print("Files in folder:")
for file_name in files_in_folder:
    file_name = file_name.split('.')[0]
    print(file_name)
    if file_name == 'gemma_finetuned_jokes_new':
        loadRating(0, 100, file_name, folder_path)
