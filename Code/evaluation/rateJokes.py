from getMetrics import getRating
import pandas as pd
import numpy as np
import os


def loadRating(start, end, file_name, folder_path, checkpoint_folder):
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
    value = 1
    for i in range(len(jokes)):
        joke = jokes[i]
        prompt = prompts[i]
        [promptScore, funnyScore, grammarScore, lengthScore, value] = getRating(joke, prompt, api_key, endpoint, value)
        print(i + 1, [promptScore, funnyScore, grammarScore, lengthScore])
        promptScores.append(promptScore)
        funnyScores.append(funnyScore)
        grammarScores.append(grammarScore)
        lengthScores.append(lengthScore)
        totalScores.append(promptScore + funnyScore + grammarScore + lengthScore)

    dictionary = {'promptScores': promptScores, 'funnyScores': funnyScores, 'grammarScores': grammarScores,
                  'lengthScores': lengthScores, 'totalScores': totalScores}

    arr_names = ['promptScores', 'funnyScores', 'grammarScores', 'lengthScores', 'totalScores']
    arr_max = [2, 2, 1, 1, 6]
    new_df = pd.DataFrame()
    for i in range(len(arr_names)):
        name = arr_names[i]
        maximum = arr_max[i]
        arr = dictionary[name]
        mean_value = np.mean(np.array(arr))
        arr.append(mean_value)
        arr.append(mean_value * 100 / maximum)
        new_df[name] = np.array(arr)

    new_df = pd.concat([df, new_df], axis=1)
    print("Rating Mean Score", totalScores[-1])
    new_df.to_csv(checkpoint_folder + file_name + '_llm_evaluation.csv', header=True, index=False)


folder_path = '../../Resources/GeneratedJokes/'
files_in_folder = os.listdir(folder_path)
checkpoint_folder = os.listdir('../../Results/LLMEvaluation/')
print("Files in folder:")
for file_name in files_in_folder:
    file_name = file_name.split('.')[0]
    print(file_name)
    if file_name + '_llm_evaluation.csv' not in checkpoint_folder:
        loadRating(0, 100, file_name, folder_path, checkpoint_folder)
