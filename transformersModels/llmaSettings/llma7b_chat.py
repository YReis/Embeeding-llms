import torch
from transformers import LlamaTokenizer, LlamaForCausalLM, LlamaModel

class LlamaModelWrapper:
    def __init__(self):
        self.chat_model_path = 'openlm-research/open_llama_7b'
        self.embedding_model_path = 'shalomma/llama-7b-embeddings'
        self.chat_tokenizer = LlamaTokenizer.from_pretrained(self.chat_model_path)
        self.chat_model = LlamaForCausalLM.from_pretrained(
            self.chat_model_path, torch_dtype=torch.float16, device_map='auto',
        )
        self.embedding_tokenizer = LlamaTokenizer.from_pretrained(self.embedding_model_path)
        self.embedding_model = LlamaModel.from_pretrained(self.embedding_model_path)

    def process_text(self, text, mode):
        if mode == 'chat':
            return self.llma7b_chat(text)
        elif mode == 'embeddings':
            return self.llama7b_embeedings(text)
        else:
            raise ValueError('Invalid mode. Expected "chat" or "embeddings".')

    def llma7b_chat(self, text):
        prompt = text
        input_ids = self.chat_tokenizer(prompt, return_tensors="pt").input_ids
        generation_output = self.chat_model.generate(
            input_ids=input_ids, max_new_tokens=32
        )

        return self.chat_tokenizer.decode(generation_output[0])

    def llama7b_embeedings(self, text):
        inputs = self.embedding_tokenizer(text, return_tensors='pt')
        outputs = self.embedding_model(**inputs)

        last_hidden_state = outputs[0]
        embedding = torch.mean(last_hidden_state, dim=1)

        return embedding.detach().numpy()
