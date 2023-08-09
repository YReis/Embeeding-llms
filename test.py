from mongo import getMongoConection
from openaiConfig import getOpenaiChat, getOpenaiEmbedding
from transformersModels import classify_text
from transformersModels import llama7bEmbeedings

client = getMongoConection()

# connect to items database
db = client["items"]
jewelItems = db["jewelItems"]
document = jewelItems.find_one()

# Loop would be here for multiple documents
# Access the 'description' field
description = document['description']
title = document['title']

# OpenAIModels

# Get embedding for the product description
# embedding = getOpenaiEmbedding(description)
# print("Embedding for the product description: ", embedding)

# Get response from OpenAI chat model


# chat_response = getOpenaiChat(f"I want you to provide me an array with categories of the following product: title: {title} , {description}, your response should be an only an array looks like this :[category1,category2,category3,category5]")
# chat_message = chat_response['choices'][0]['message']['content']
# print("Response from OpenAI Chat model: ", chat_message)

# TransformersModels

# Get response from classify_text function
classify_response = classify_text(f"I want you to provide me an array with categories of the following product: title: {title} , {description}, your response should be an only an array looks like this :[category1,category2,category3,category5]")
print("Response from classify_text function: ", classify_response)
classify_response1 = classify_text1(f"I want you to provide me an array with categories of the following product: title: {title} , {description}, your response should be an only an array looks like this :[category1,category2,category3,category5]")
print("Response from classify_text function: ", classify_response1)