# 685_Joke_Generator
UMass Amherst CS-685 Project for Text-based Joke Generation Large Language Models (LLMs)

## Code Structure:
- Code Folder: Contains all python code for this project
  - promptCreation (createPrompt.py) : Contains code to create prompts from shortjokes dataset and create jokes dataset.
  - evaluation (rateJokes.py) : Contains code to rate LLM generated jokes based on promptMatching, Funniness, Grammar and Length.
  - evaluation (evalJokes.py) : Contains code to compare 2 jokes and give the better one for each model.
  - training (CS685_Joke_Generator_LoRA+DPO.ipynb) : Contains code to train models on training data, and generate LoRA-only test results and DPO training data (Note: This notebook runs on Colab)

- Resources Folder: Contains all required datasets.
  - Dataset: Contains the original shortjokes dataset along with the generated training, dpo and testing datasets.
  - GeneratedJokes: Contains all LLM generated jokes for all models (pretrained, finetuned and dpo).
  - CombinedJokes: Contains a single dataset of all generated jokes per model.
  
- Results Folder: Contains all generated results.
  - HumanEvaluation: Contains all human evaluated datasets of all LLM generated jokes.
  - LLMEvaluation: Contains all LLM evaluated datasets of all LLM generated jokes.
  - RankScoreEvaluation: Contains the comparison evaluation of all the combined LLM datasets.


## Setup:
- Requires python3 to be installed.
- Requires Matplotlib, Pandas, Numpy, SkLearn and Requests to be installed.
- Python version used to test is Python 3.11.5

## RunCode:
- createPrompt.py : Run this file to create prompts and generate training dataset.
- rateJokes.py : Run this file to get LLM scores for the LLM generated jokes on promptMatching.
- evalJokes.py : Run this file to get comparison scores on combined dataset of each LLM.

