class InferenceService:

    def __init__(self, engine):
        self.engine = engine

    def generate(self, prompt: str, max_new_tokens: int = 100):
        return self.engine.submit(
            prompt,
            max_new_tokens=max_new_tokens,
        )