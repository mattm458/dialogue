from random import sample
import tempfile

import speech_recognition as sr


class SRAudioIn:
    def __init__(self, device_index, sample_rate):
        self.device_index = device_index
        self.sample_rate = sample_rate

        self.r = sr.Recognizer()

    def __call__(self, context):
        with sr.Microphone(
            device_index=self.device_index, sample_rate=self.sample_rate
        ) as source:
            print("Listening...")
            self.r.adjust_for_ambient_noise(source)
            audio = self.r.listen(source)

            context.add_user_audio(audio)
