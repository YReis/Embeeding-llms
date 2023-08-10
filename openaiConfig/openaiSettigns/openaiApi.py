import openai
import os
from dotenv import load_dotenv

def getOpenaiChat(text):
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")
    
    chat_completion = openai.ChatCompletion.create(model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": f"{text}"}])

    return(chat_completion)

def getOpenaiEmbedding(text, model="text-embedding-ada-002"):
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")
    text = text.replace("\n", " ")
    return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']

#combination of both

def getOpenaiEmbeddingAndChat(message,mode):
    if mode == 'chat':
        return getOpenaiChat(message)
    elif mode == 'embedding':
        return getOpenaiEmbedding(message)
    else:
        return "mode not supported, please choose between 'chat' or 'embedding'"

    