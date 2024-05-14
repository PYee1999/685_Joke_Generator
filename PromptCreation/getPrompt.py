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


def getPrompt(joke, api_key, endpoint, model='mistralai/Mixtral-8x7B-Instruct-v0.1'):
    prompt = """
    INSTRUCTIONS: Given below is a list of jokes along with their prompts. Generate a similar prompt for the joke I gave. Only give me the prompt, do not have 'Prompt: ' in it.
    
    Joke: [me narrating a documentary about narrators] I can't hear what they're saying cuz I'm talking
    Prompt: Tell me a joke about a documentary.
    
    Joke: Telling my daughter garlic is good for you. Good immune system and keeps pests away.Ticks, mosquitos, vampires... men.
    Prompt: Give me a funny quip about parents and kids.
    
    Joke: If I could have dinner with anyone, dead or alive... ...I would choose alive.
    Prompt: Can you craft a joke about dinner?
        
    Joke: 'Thought I saw a walking burrito but it was just a pug in a raincoat.'
    Prompt: Make me laugh with a joke about a pug.
    
    Joke: {} 
    Prompt: {}
    
    """.format(joke, '')

    # Generate text
    generated_text = generate_text(prompt, api_key, endpoint, model)
    return generated_text

