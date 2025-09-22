from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch


class LLMHandler:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large").to(self.device)

    def infer(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs, max_new_tokens=300)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
