class TerminalASR:
    def __call__(self, context):
        in_text = input(">> ")
        context.add_user_text(in_text)
        return in_text