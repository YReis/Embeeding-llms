from flask import Flask, request, jsonify
from openaiConfig import getOpenaiEmbeddingAndChat
from transformersModels import LlamaModelWrapper

app = Flask(__name__)

llama = LlamaModelWrapper()

@app.route('/api/getOpenaiChat', methods=['POST'])
def api_getOpenaiContent():
    data = request.get_json()
    mode = data.get('mode')
    title = data.get('title')
    description = data.get('description')
    
    response = getOpenaiEmbeddingAndChat(f"{title}{description}", mode)
    return jsonify({"response": response})


@app.route('/api/llama', methods=['POST'])
def api_llama():
    data = request.get_json()
    title = data.get('title')
    description = data.get('description')
    mode = data.get('mode')
    response = llama.process_text(f"{title}{description}", mode)
    if mode == 'embeddings':
        response = response.tolist() 
    return jsonify({"response": response})


if __name__ == '__main__':
    app.run(debug=True)
