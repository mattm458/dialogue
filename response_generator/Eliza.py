from response_generator.eliza import eliza

class ElizaResponseGenerator:
    def __init__(self):
        self.eliza = eliza.Eliza()
        self.eliza.load("response_generator/eliza/doctor.txt")

    def __call__(self, context):
        response = self.eliza.respond(context.get_latest_user_text())
        context.add_response_text(response)
        print(f'Eliza: {response}')
        return response