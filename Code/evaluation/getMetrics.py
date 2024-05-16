import requests


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


def getFunnyScore(joke, value=1):
    API_TOKEN = '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'
    API_TOKEN2 = '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'
    API_TOKEN3 = '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'
    API_TOKEN4 = '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'
    API_URL = "https://api-inference.huggingface.co/models/mohameddhiab/humor-no-humor"
    for i in range(4):
        try:
            if value == 2:
                API_TOKEN = API_TOKEN2
            if value == 3:
                API_TOKEN = API_TOKEN3
            if value == 4:
                API_TOKEN = API_TOKEN4
            headers = {"Authorization": f"Bearer {API_TOKEN}"}
            response = requests.post(API_URL, headers=headers, json=joke)
            response.raise_for_status()  # Check for any HTTP errors
            humour = response.json()
            if humour[0][0]['label'] == 'HUMOR' and humour[0][0]['score'] >= 0.9:
                return [2, value]
            elif humour[0][0]['label'] == 'HUMOR' and humour[0][0]['score'] < 0.9:
                return [1, value]
            else:
                return [0, value]
        except requests.exceptions.RequestException as e:
            print(e)
            value += 1
    print("Timed Out")
    return [-1, 1]


def getFunnyScoreLLM(joke, api_key, endpoint, model):
    Instruction = """
    INSTRUCTIONS: Given a joke, rate the joke based on the below rubric.
    a. give score 2 if the joke is very funny.
    b. give score 1 if the joke is slightly funny.
    c. give score 0 if the joke is not funny.
    
    Only return the score value, do not give any explanation. See the examples below.
    
    Joke: [me narrating a documentary about narrators] I can't hear what they're saying cuz I'm talking.
    Score: 2
    
    Joke You can't change the past. But you can sit around in your underwear, dwelling on it and crying over what could have been.
    Score: 2
    
    Joke:  My husband told me the other day that I should wear more perfume to attract men. I guess he's planning on moving into the kitchen soon. 
    Score: 1
    
    Joke:  Funny mathematicians don't calculate very well. They estimate and round up.<eos>
    Score: 1
    
    Joke: This is a joke about pencil.
    Score: 0
    
    Joke: Why did the job interviewer eat his coat? He said he needed a coat of paint.<eos>
    Score: 0
    
    Generate a similar Score for the Joke I gave. Follow the rubric exactly.
    Only give me the Score and nothing else, do not have 'Score: ' in it.
    
    Joke: {} 
    Score: {}
    """.format(joke, '')

    # Generate text
    score = generate_text(Instruction, api_key, endpoint, model)
    return extract_first_integer(score)


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


def getRating(joke, prompt, api_key, endpoint, value, model='meta-llama/Llama-3-70b-chat-hf',):
    promptScore = getPromptScore(joke, prompt, api_key, endpoint, model)
    [funnyScore, value] = getFunnyScore(joke, value)
    if funnyScore == -1:
        funnyScore = getFunnyScoreLLM(joke, api_key, endpoint, model)
    grammarScore = getGrammarScore(joke, api_key, endpoint, model)
    lengthScore = 0
    if len(joke) < 100:
        lengthScore = 1
    '''
    print('Prompt', promptScore)
    print('Funny', funnyScore)
    print('Grammar', grammarScore)
    '''
    return [promptScore, funnyScore, grammarScore, lengthScore, value]


def getEval(prompt, preJoke, fineJoke, api_key, endpoint, model='mistralai/Mixtral-8x7B-Instruct-v0.1'):
    Instruction = """INSTRUCTIONS: Given a prompt and 2 jokes (PreJoke and FineJoke), return the better joke. return 
    only the label of the better joke. Do not give explanation.
    
    Prompt: Give me a funny parent-child joke?
    PreJoke: Telling my daughter garlic is good for you. Good immune system and keeps pests away.Ticks, mosquitos, vampires... men.
    FineJoke: Shush you! I gave birth to you. 
    Answer: PreJoke
    
    Prompt: Tell me a joke about a documentary?
    PreJoke: narrators are narrating storytellers.
    FineJoke: [me narrating a documentary about narrators] I can't hear what they're saying cuz I'm talking
    Answer: FineJoke
    
    Prompt: Can you come up with a joke about dolphins?
    PreJoke: What do you call a dolphin who is sick? A sick dolphin.
    FineJoke: Do you know about those people who take pictures of dolphins that look like naked people? Well, they're called whale-fashion photographers.
    Answer: FineJoke
    
    Prompt: Make me laugh with a joke about the DMV.
    PreJoke:  I think the D.M.V. makes us all feel like that little boy from the movie "A Nightmare on Elm Street". Every time you go there, you think... GGggggh! Wake me up! Wake me up!!!
    FineJoke: What do you call a person who has been waiting in line at the DMV for 10 years?
    Answer: PreJoke
    
    Prompt: {}
    PreJoke: {} 
    FineJoke: {}
    Answer: {}
    
    """.format(prompt, preJoke, fineJoke, '')

    score = generate_text(Instruction, api_key, endpoint, model)
    try:
        score = score.split(' ')[1]
    except:
        pass
    if score == 'FineJoke':
        return 1
    if score == 'PreJoke':
        return -1
    return 0
