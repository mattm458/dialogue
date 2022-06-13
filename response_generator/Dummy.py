class DummyResponseGenerator:
    def __init__(self, response="Testing"):
        self.response = response

    def __call__(self, context):
        context.add_response_text(self.response)
        print(f"Dummy: {self.response}")
