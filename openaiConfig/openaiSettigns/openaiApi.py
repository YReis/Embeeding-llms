import openai
import os
from dotenv import load_dotenv

def getOpenaiChat(message):
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")
    
    chat_completion = openai.ChatCompletion.create(model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": f"{message}"}])

    return(chat_completion)

def getOpenaiEmbedding(text, model="text-embedding-ada-002"):
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")
    text = text.replace("\n", " ")
    return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']