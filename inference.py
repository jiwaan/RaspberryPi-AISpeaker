from typing import Any


class Inference():
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor
    
    def __call__(self):
        return 