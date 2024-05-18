import os

from getMetrics import getEval
import pandas as pd
import numpy as np


def loadEval(start, end, file_name, folder_path, checkpoint_folder):
    df_total = pd.read_csv(folder_path + file_name + '.csv')
    df = df_total.iloc[start:end].copy()
    prompts = df['Prompts'].values.flatten()
    dpo_jokes = df['DPOJokes'].values.flatten()
    pretrained_jokes = df['PretrainedJokes'].values.flatten()
    finetuned_jokes = df['FinetunedJokes'].values.flatten()
    scoresFinetuned = []
    scoresDPO = []
    scoresDPO2 = []
    api_key = '' # Add api key here
    endpoint = 'https://api.together.xyz/v1/chat/completions'
    for i in range(len(prompts)):
        prompt = prompts[i]
        dpo_joke = dpo_jokes[i]
        pretrained_joke = pretrained_jokes[i]
        finetuned_joke = finetuned_jokes[i]
        scoreFinetuned = getEval(prompt, pretrained_joke, finetuned_joke, api_key, endpoint)
        scoreDPO = getEval(prompt, pretrained_joke, dpo_joke, api_key, endpoint)
        scoreDPO2 = getEval(prompt, finetuned_joke, dpo_joke, api_key, endpoint)
        print(prompt, scoreFinetuned, scoreDPO, scoreDPO2)
        scoresFinetuned.append(scoreFinetuned)
        scoresDPO.append(scoreDPO)
        scoresDPO2.append(scoreDPO2)

    df['SFTvsPre'] = np.array(scoresFinetuned)
    df['DPOvsPre'] = np.array(scoresDPO)
    df['DPOvsSFT'] = np.array(scoresDPO2)
    df.to_csv(checkpoint_folder + file_name + '_rs_evaluation.csv', header=True, index=False)


folder_path = '../../Resources/CombinedJokes/'
files_in_folder = os.listdir(folder_path)
checkpoint_folder = '../../Results/RankScoreEvaluation/'
print("Files in folder:")
for file_name in files_in_folder:
    file_name = file_name.split('.')[0]
    print(file_name)
    if file_name + '_rs_evaluation.csv' not in os.listdir(checkpoint_folder):
        loadEval(0, 100, file_name, folder_path, checkpoint_folder)
