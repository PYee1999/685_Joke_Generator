import pandas as pd
import os
import numpy as np
from sklearn.metrics import confusion_matrix


def cohen_kappa(y1, y2):
    cm = confusion_matrix(y1, y2)
    n = np.sum(cm)
    Po = np.trace(cm) / n
    sum_rows = np.sum(cm, axis=1)
    sum_cols = np.sum(cm, axis=0)
    Pe = np.sum(sum_rows * sum_cols) / (n * n)
    kappa = (Po - Pe) / (1 - Pe)

    return kappa


def getCohenKappa(humfile, llmfile):
    humdf = pd.read_csv('../../Results/HumanEvaluation/' + humfile + '.csv')
    llmdf = pd.read_csv('../../Results/LLMEvaluation/' + llmfile + '.csv')
    ck = 0
    for i in range(2, 5):
        a = humdf.iloc[:, i].values[0:100].astype(int).tolist()
        b = llmdf.iloc[:, i].values[0:100].astype(int).tolist()
        ck += cohen_kappa(a, b)
    print("Cohen's Kappa", ck/3)


human_path = '../../Results/HumanEvaluation'
llm_path = '../../Results/LLMEvaluation'
files_in_folder = os.listdir(human_path)
print("Files in folder:")
for file_name in files_in_folder:
    human_file = file_name.split('.')[0]
    llm_file = human_file[:-17] + '_llm_evaluation'
    print(human_file, llm_file)
    getCohenKappa(human_file, llm_file)
