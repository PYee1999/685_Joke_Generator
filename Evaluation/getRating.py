import requests


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


def getRating(joke, prompt, api_key, endpoint, model='mistralai/Mixtral-8x7B-Instruct-v0.1'):
    Instruction = """
    Prompt: {}
    Joke: {} 
    INSTRUCTIONS: Given a prompt and joke, give likert scale rating for it indicating how funny it is and how well it matches the joke. Give just the rating.
    
    Joke: [me narrating a documentary about narrators] I can't hear what they're saying cuz I'm talking
    Prompt: Tell me a joke about a documentary?
    Rating: 3/5
    
    Joke: Telling my daughter garlic is good for you. Good immune system and keeps pests away.Ticks, mosquitos, vampires... men.
    Prompt: Give me a funny parent-child joke?
    Rating: 4/5
    
    Generate a similar rating for the joke and prompt I gave. Only give me the Rating, do not have 'Rating: ' in it.
    """.format(prompt, joke)

    print(Instruction)

    # Generate text
    generated_text = generate_text(Instruction, api_key, endpoint, model)
    return generated_text

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
