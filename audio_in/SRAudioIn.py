import tempfile

import speech_recognition as sr


class SRAudioIn:
    def __init__(self):
        self.r = sr.Recognizer()

    def __call__(self, context):
        with sr.Microphone(device_index=16, sample_rate=32000) as source:
            print("Listening...")
            self.r.adjust_for_ambient_noise(source)
            audio = self.r.listen(source)

            context.add_user_audio(audio)
