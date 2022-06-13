import speech_recognition as sr

class SphinxASR:
    def __init__(self):
        self.r = sr.Recognizer()

    def __call__(self, context):
        audio = context.get_latest_user_audio()
        text = self.r.recognize_sphinx(audio)
        context.add_user_text(text)

        print(f'ASR: Heard "{text}"')

        return text
