import tempfile

import hyphenate
import numpy as np
import parselmouth
import pyttsx3
import soundfile as sf
import speech_recognition as sr
from nltk.corpus import cmudict

# from transformers import AutoModelForCausalLM, AutoTokenizer
from eliza import eliza

d = cmudict.dict()


def nsyl(word):
    """Count the number of syllables in a word."""
    # Remove trailing whitespace and convert to lowercase for dictionary lookup.
    word = word.strip().lower()

    # Special case: Empty string.
    if len(word) == 0:
        return 0

    # Special case: If there is an apostrophe in the word, then it may not be
    # in the dictionary.
    if "'" in word:
        # A common situation is "'s", where the dictionary does not contain the possessive
        # form of all words. If that applies here, remove the "'s" and look up the
        # singular form of the word.
        if word not in d and word[-2:] == "'s":
            word = word[:-2]

    # Main syllable lookup functionality.
    if word in d:
        # If the word is in the dictionary, extract the syllable count.
        return [len(list(y for y in x if y[-1].isdigit())) for x in d[word]][0]
    else:
        # Otherwise, fall back to the hyphenate library for a best (but
        # sometimes inaccurate) guess.
        return len(hyphenate.hyphenate_word(word))


class Context:
    def __init__(self):
        self.user_audios = []
        self.user_texts = []
        self.response_texts = []
        self.user_pitch = []
        self.user_duration = []
        self.user_rate = []

    def add_user_audio(self, user_audio):
        self.user_audios.append(user_audio)

    def add_user_text(self, user_text):
        self.user_texts.append(user_text)

    def add_response_text(self, response_text):
        self.response_texts.append(response_text)

    def add_user_pitch(self, user_pitch):
        self.user_pitch.append(user_pitch)

    def add_user_duration(self, user_duration):
        self.user_duration.append(user_duration)

    def add_user_rate(self, user_rate):
        self.user_rate.append(user_rate)

    def get_latest_user_audio(self):
        return self.user_audios[-1]

    def get_latest_user_text(self):
        return self.user_texts[-1]

    def get_latest_response_text(self):
        return self.response_texts[-1]

    def get_latest_user_pitch(self):
        return self.user_pitch[-1]

    def get_latest_user_duration(self):
        return self.user_duration[-1]

    def get_latest_user_rate(self):
        return self.user_rate[-1]

    def get_text_history(self):
        hist = []
        for i in range(len(self.response_texts)):
            hist.append(self.user_texts[i])
            hist.append(self.response_texts[i])

        hist.append(self.user_texts[-1])

        return hist


# class DialoGPTResponseGenerator:
#     def __init__(self, size="large"):
#         self.tokenizer = AutoTokenizer.from_pretrained(f"microsoft/DialoGPT-{size}")
#         self.model = AutoModelForCausalLM.from_pretrained(
#             f"microsoft/DialoGPT-{size}"
#         ).cuda()

#     def __call__(self, context):
#         history = (
#             self.tokenizer.eos_token.join(context.get_text_history())
#             + self.tokenizer.eos_token
#         )

#         gpt_input = self.tokenizer.encode(history, return_tensors="pt").cuda()

#         chat_history_ids = self.model.generate(
#             gpt_input,
#             max_length=1000,
#             temperature=0.6,
#             repetition_penalty=1.3,
#             pad_token_id=self.tokenizer.eos_token_id,
#         )

#         return self.tokenizer.decode(
#             chat_history_ids[:, gpt_input.shape[-1] :][0], skip_special_tokens=True
#         )


class ElizaResponseGenerator:
    def __init__(self):
        self.eliza = eliza.Eliza()
        self.eliza.load("eliza/doctor.txt")

    def __call__(self, context):
        response = self.eliza.respond(context.get_latest_user_text())
        context.add_response_text(response)
        print(response)


class DummyResponseGenerator:
    def __call__(self, context):
        text = "testing"
        print(text)
        context.add_response_text(text)


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


class DummyTerminalASR:
    def __call__(self, context):
        in_text = input(">> ")
        context.add_user_text(in_text)


class DummyASR:
    def __call__(self, context):
        in_text = "Testing"
        context.add_user_text(in_text)


class DummyAudioIn:
    def __call__(self, context):
        pass


class ASR:
    def __init__(self):
        self.r = sr.Recognizer()

    def __call__(self, context):
        audio = context.get_latest_user_audio()
        text = self.r.recognize_sphinx(audio)

        duration = context.get_latest_user_duration()
        rate = sum([nsyl(w) for w in text.split(" ")]) * 60 / duration
        print("> " + text)
        context.add_user_text(text)
        context.add_user_rate(rate)


class SRAudioIn:
    def __init__(self):
        self.r = sr.Recognizer()

    def __call__(self, context):
        with sr.Microphone(device_index=16, sample_rate=32000) as source:
            self.r.adjust_for_ambient_noise(source)
            audio = self.r.listen(source)
            wav_data = audio.get_wav_data()

            with tempfile.NamedTemporaryFile() as tmp:
                tmp.write(wav_data)
                sound = parselmouth.Sound(tmp.name)

            mean_pitch = parselmouth.praat.call(
                sound.to_pitch(), "Get mean", 0, 0, "Hertz"
            )
            duration = sound.duration
            context.add_user_pitch(mean_pitch)
            context.add_user_audio(audio)
            context.add_user_duration(duration)


if __name__ == "__main__":
    print("Starting")
    audio_in = SRAudioIn()
    asr = ASR()
    response_generator = ElizaResponseGenerator()
    tts = PyTTSX3()

    context = Context()

    for i in range(5):
        audio_in(context)
        asr(context)
        response_generator(context)
        tts(context)

    print(context.get_text_history())
