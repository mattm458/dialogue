import pyttsx3

class PyTTSX3:
    def __init__(self):
        self.engine = pyttsx3.init()

    def __call__(self, context):
        pitch = context.get_latest_user_pitch()
        rate = context.get_latest_user_rate() / 1.5
        pitch /= 2
        self.engine.setProperty("rate", rate)
        self.engine.setProperty("pitch", pitch)
        self.engine.say(context.get_latest_response_text())
        self.engine.runAndWait()