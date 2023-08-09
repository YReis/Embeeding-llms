from flask import Flask, request, jsonify
from openaiConfig import getOpenaiChat, getOpenaiEmbedding
from transformersModels import LlamaModelWrapper

app = Flask(__name__)

llama = LlamaModelWrapper()

@app.route('/api/getOpenaiChat', methods=['POST'])
def api_getOpenaiChat():
    data = request.get_json()
    title = data.get('title')
    description = data.get('description')
    chat_response = getOpenaiChat(f"I want you to provide me an array with categories of the following product: title: {title} , {description}, your response should be an only an array looks like this :[category1,category2,category3,category5]")
    chat_message = chat_response['choices'][0]['message']['content']
    return jsonify({"chat_message": chat_message})


@app.route('/api/getOpenaiEmbedding', methods=['POST'])
def api_getOpenaiEmbedding():
    data = request.get_json()
    description = data.get('description')
    embedding = getOpenaiEmbedding(description)
    return jsonify({"embedding": embedding.tolist()})


@app.route('/api/llama', methods=['POST'])
def api_llama():
    data = request.get_json()
    title = data.get('title')
    description = data.get('description')
    mode = data.get('mode')
    response = llama.process_text(f"{title}{description}", mode)
    if mode == 'embeddings':
        response = response.tolist()  # Convert ndarray to list
    return jsonify({"response": response})


if __name__ == '__main__':
    app.run(debug=True)
