import requests
import time

def extract_first_integer(sentence):
    words = sentence.split()
    for word in words:
        try:
            number = int(word)
            return number
        except ValueError:
            continue
    return 0


def generate_text(prompt, api_key, endpoint, model):
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }

    data = {
        'model': model,
        'messages': [
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': prompt}
        ],
        # 'max_tokens': 2048,  # You can set a maximum token limit
        'temperature': 0.7,
        'top_p': 0.5,
        'frequency_penalty': 0.0,
        'presence_penalty': 0.0,
        # 'stop': ["\n", ".", "!", "?"],  # Tokens to stop at end of sentences
        'n': 1
    }
    response = requests.post(endpoint, headers=headers, json=data)
    # print("response : ", response.json())
    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    else:
        print("Error:", response.text)
        return None


def getFunnyScore(joke):
    API_TOKEN = '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'
    API_URL = "https://api-inference.huggingface.co/models/mohameddhiab/humor-no-humor"
    headers = {"Authorization": f"Bearer {API_TOKEN}"}
    for _ in range(3):
        try:
            response = requests.post(API_URL, headers=headers, json=joke)
            response.raise_for_status()  # Check for any HTTP errors
            humour = response.json()
            if humour[0][0]['label'] == 'HUMOR' and humour[0][0]['score'] >= 0.9:
                return 2
            elif humour[0][0]['label'] == 'HUMOR' and humour[0][0]['score'] < 0.9:
                return 1
            else:
                return 0
        except requests.exceptions.RequestException as e:
            pass
    return 0


'''
def getFunnyScore(joke, api_key, endpoint, model):
    Instruction = """
    INSTRUCTIONS: Given a joke, rate the joke based on the below rubric.
    a. give score 2 if the joke is very funny.
    b. give score 1 if the joke is slightly funny.
    c. give score 0 if the joke is not funny.
    
    Only return the score value, do not give any explanation. See the examples below.
    
    Joke: [me narrating a documentary about narrators] I can't hear what they're saying cuz I'm talking.
    Score: 2
    
    Joke: dark dark go away. 
    Score: 1
    
    Joke: This is a joke about pencil.
    Score: 0
    
    Generate a similar Score for the Joke I gave. Follow the rubric exactly.
    Only give me the Score and nothing else, do not have 'Score: ' in it.
    
    Joke: {} 
    Score: {}
    """.format(joke, '')

    # Generate text
    score = generate_text(Instruction, api_key, endpoint, model)
    return extract_first_integer(score)
'''


def getGrammarScore(joke, api_key, endpoint, model):
    Instruction = """
    INSTRUCTIONS: Given a sentence, rate the sentence based on the below rubric.
    a. give score 1 if the sentence is grammatically correct.
    b. give score 0 if the sentence is not grammatically correct.
    
    Only return the score value, do not give any explanation. See the examples below.
    
    Sentence: This is a grammatically correct sentence.
    Score: 1
    
    Sentence: This are not one grammar correct. 
    Score: 0
    
    Generate a similar Score for the Sentence I gave. Follow the rubric exactly.
    Only give me the Score and nothing else, do not have 'Score: ' in it.
    
    Sentence: {} 
    Score: {}
    """.format(joke, '')

    # Generate text
    score = generate_text(Instruction, api_key, endpoint, model)
    return extract_first_integer(score)


def getPromptScore(joke, prompt, api_key, endpoint, model):
    Instruction = """
    INSTRUCTIONS: Given a prompt and joke, rate the joke based on the below rubric.
    a. give score 2 if the joke directly corresponds to the prompt.
    b. give score 1 if the joke somewhat relates to the prompt.
    c. give score 0 if the joke does not align with the prompt.
    
    Only return the score value, do not give any explanation. See the examples below.
    
    Joke: [me narrating a documentary about narrators] I can't hear what they're saying cuz I'm talking
    Prompt: Tell me a joke about a documentary?
    Score: 2
    Reason: The joke is about narrators narrating a documentary which corresponds to the given prompt.
    
    Joke: is it light outside? 
    Prompt: Give me a dark joke?
    Score: 1
    Reason: This is somewhat ok as light relates to dark.
    
    Joke: why did the chicken cross the road? it wanted to go to the other side.
    Prompt: Give me a joke about a pencil?
    Score: 0
    Reason: Joke does not align with prompt as it has nothing to do with a pencil
    
    Generate a similar Score for the Joke and Prompt I gave. Follow the rubric exactly.
    Only give me the Score and nothing else, do not have 'Score: ' in it. Do not give me Reason.
    
    Prompt: {}
    Joke: {} 
    Score: {}
    """.format(prompt, joke, '')

    # Generate text
    score = generate_text(Instruction, api_key, endpoint, model)
    return extract_first_integer(score)


def getRating(joke, prompt, api_key, endpoint, model='meta-llama/Llama-3-70b-chat-hf'):
    promptScore = getPromptScore(joke, prompt, api_key, endpoint, model)
    funnyScore = getFunnyScore(joke)
    grammarScore = getGrammarScore(joke, api_key, endpoint, model)
    lengthScore = 0
    if len(joke) < 100:
        lengthScore = 1
    '''
    print('Prompt', promptScore)
    print('Funny', funnyScore)
    print('Grammar', grammarScore)
    '''
    return [promptScore, funnyScore, grammarScore, lengthScore]


def getEval(prompt, preJoke, fineJoke, api_key, endpoint, model='mistralai/Mixtral-8x7B-Instruct-v0.1'):
    Instruction = """
    Prompt: {}
    Pretrained Joke: {} 
    Finetuned Joke: {}
    INSTRUCTIONS: Given a prompt and 2 jokes (preJoke and fineJoke), return the joke which is funnier and matches the prompt well.
    
    Prompt: Tell me a joke about a documentary?
    Pretrained Joke: [me narrating a documentary about narrators] I can't hear what they're saying cuz I'm talking
    FineTuned Joke: narrators are narrating storytellers.
    Best Joke: Pretrained
    
    Joke: Telling my daughter garlic is good for you. Good immune system and keeps pests away.Ticks, mosquitos, vampires... men.
    Prompt: Give me a funny parent-child joke?
    Pretrained Joke: Shush you! I gave birth to you.
    Finetuned Joke: Telling my daughter garlic is good for you. Good immune system and keeps pests away.Ticks, mosquitos, vampires... men.
    Best Joke: Finetuned
    
    Return a similar Best Joke for the prompt and jokes I gave. Only give me the Label, do not have 'Best Joke: ' in it.
    """.format(prompt, preJoke, fineJoke)

    print(Instruction)

    # Generate text
    generated_text = generate_text(Instruction, api_key, endpoint, model)
    return generated_text
