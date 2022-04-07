from context import Context
from audio_in.SRAudioIn import SRAudioIn
from asr.SRASR import SRASR
from response_generator.ElizaResponseGenerator import ElizaResponseGenerator
from tts.PyTTSX3TTs import PyTTSX3
from tts.Tacotron2TTS import Tacotron2TTS
from feature_extractor.PraatFeatureExtractor import PraatFeatureExtractor
from feature_extractor.RateFeatureExtractor import RateFeatureExtractor

if __name__ == "__main__":
    print("Starting")
    audio_in = SRAudioIn()
    asr = SRASR()
    response_generator = ElizaResponseGenerator()
    tts = Tacotron2TTS() #PyTTSX3()

    praat_feature_extractor = PraatFeatureExtractor()
    rate_feature_extractor = RateFeatureExtractor()

    context = Context()

    for i in range(5):
        audio_in(context)
        asr(context)

        praat_feature_extractor(context)
        rate_feature_extractor(context)

        response_generator(context)
        tts(context)

    print(context.get_text_history())
