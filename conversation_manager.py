from collections import deque
from config import CONFIG

class ConversationManager:
    def __init__(self, max_history=CONFIG["max_conversation_history"]):
        self.history = deque(maxlen=max_history)
    
    def add_entry(self, question, answer):
        self.history.append({"question": question, "answer": answer})
    
    def get_history(self):
        # Menggabungkan history menjadi string untuk context QA
        return [f"{entry['question']}\n{entry['answer']}" for entry in self.history]
