import argparse
from urllib import response

from regex import E

from context import Context

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A dialogue engine for entrainment research in human-computer conversations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    general_parser = parser.add_argument_group("General options")
    general_parser.add_argument(
        "--max-turns",
        required=False,
        default=5,
        help="The number of conversation turns to engage in before quitting.",
        type=int,
    )

    asr_parser = parser.add_argument_group("Automatic speech recognition")
    asr_parser.add_argument(
        "--asr",
        required=False,
        type=str,
        default="terminal",
        help="The automatic speech recognition module to use for input.",
        choices=["terminal", "sphinx"],
    )

    audio_in_parser = parser.add_argument_group("Audio in")
    audio_in_parser.add_argument(
        "--audio-in",
        required=False,
        type=str,
        default="dummy",
        help="The module to use for recording audio data.",
        choices=["dummy", "sr"],
    )
    audio_in_parser.add_argument(
        "--sr-audio-in-device-index",
        required=False,
        type=int,
        help="The index of the audio device to use for recording in the SR audio in module. Required if --audio-in=sr.",
    )
    audio_in_parser.add_argument(
        "--sr-audio-in-sample-rate",
        required=False,
        type=int,
        help="The sample rate to use for recording in the SR audio in module.",
        default=32000,
    )

    entrainment_strategy_parser = parser.add_argument_group("Entrainment strategy")
    entrainment_strategy_parser.add_argument(
        "--entrainment-strategy",
        required=False,
        type=str,
        default="neutral",
        help="The entrainment strategy to use in determining speech acoustic/prosodic features.",
        choices=["neutral", "matching", "neural"],
    )

    entrainment_strategy_parser.add_argument(
        "--neural-entrainment-ckpt",
        required=False,
        help="The path to a trained neural entrainment checkpoint. Required if --entrainment-strategy=neural.",
    )

    feature_extractor_parser = parser.add_argument_group("Feature extractor")
    feature_extractor_parser.add_argument(
        "--feature-extractor",
        required=False,
        type=str,
        help="The feature extractors to use on recorded input.",
        choices=["dummy", "praat", "rate"],
        nargs="+",
    )
    feature_extractor_parser.add_argument(
        "--dummy-extractor-config",
        required=False,
        default=None,
        help="The path to a JSON file containing hardcoded feature values to always output. Required with the dummy feature extractor.",
        type=str,
    )
    feature_extractor_parser.add_argument(
        "--dummy-extractor-fuzz",
        required=False,
        default=0.0,
        help="A percentage to randomly fuzz hardcoded feature values in either direction. Default value is 0.0.",
        type=float,
    )

    response_generator_parser = parser.add_argument_group("Response generator")
    response_generator_parser.add_argument(
        "--response-generator",
        required=False,
        type=str,
        default="eliza",
        help="The system to use for determining conversational responses.",
        choices=["dummy", "eliza", "dialogpt"],
    )
    response_generator_parser.add_argument(
        "--dummy-response-generator-response",
        required=False,
        type=str,
        default="Testing",
        help="The string to always respond with. Used with the dummy response generator.",
    )
    response_generator_parser.add_argument(
        "--dialogpt-size",
        required=False,
        type=str,
        default="medium",
        help="The DialoGPT model size to download. Used with the dialogpt response generator.",
        choices=["small", "medium", "large"],
    )

    tts_parser = parser.add_argument_group("Text-to-speech")
    tts_parser.add_argument(
        "--tts",
        required=False,
        default="terminal",
        help="The TTS engine to use for producing speech output.",
        choices=["dummy", "pyttsx3", "tacotron2"],
    )
    tts_parser.add_argument(
        "--tacotron2-device-index",
        required=False,
        type=int,
        help="The index of the audio device to use for outputting TTS audio. Required if --tts=tacotron2.",
    )
    tts_parser.add_argument(
        "--tacotron2-ckpt",
        required=False,
        help="The path to a trained Tacotron 2 model checkpoint. Required if --tts=tacotron2.",
    )

    transformer_parser = parser.add_argument_group(
        "Transform recorded values for an entrainment strategy."
    )
    transformer_parser.add_argument(
        "--transformers",
        required=False,
        default=["passthrough"],
        help="Transform modules to invoke prior to passing final entrained values to the TTS.",
        choices=["passthrough", "pitch_range", "log", "normalize", "normalize_user"],
        nargs="+",
    )
    transformer_parser.add_argument(
        "--normalize-transformer-config",
        required=False,
        default=None,
        help="The path to a JSON file containing a vocal range to normalize within. Required for the Normalize transformer.",
        type=str,
    )

    entrainment_transformer_parser = parser.add_argument_group(
        "Transform entrained values for the TTS."
    )
    entrainment_transformer_parser.add_argument(
        "--entrainment-transformers",
        required=False,
        default=["passthrough"],
        help="Transform modules to invoke prior to passing final entrained values to the TTS.",
        choices=["passthrough", "map", "expand"],
        nargs="+",
    )
    entrainment_transformer_parser.add_argument(
        "--map-transformer-target-config",
        required=False,
        default=None,
        help="The path to a JSON file containing the target vocal range. Required for the Map transformer.",
        type=str,
    )
    entrainment_transformer_parser.add_argument(
        "--map-transformer-speaker-config",
        required=False,
        default=None,
        help="The path to a JSON file containing the speaker vocal range. Optional for the Map transformer. If not supplied, the Map transformer will use a neutral tone until a suitable range can be determined from recorded vocal samples. If given, the map transformer will use the speaker config until a suitable range can be determined from vocal samples.",
        type=str,
    )
    entrainment_transformer_parser.add_argument(
        "--expand-transformer-target-config",
        required=False,
        default=None,
        help="The path to a JSON file containing the target vocal range. Required for the Expand transformer.",
        type=str,
    )

    args = parser.parse_args()

    print(args)
    print("Loading...")

    if args.asr == "terminal":
        from asr.Terminal import TerminalASR

        asr = TerminalASR()
    elif args.asr == "sphinx":
        from asr.Sphinx import SphinxASR

        asr = SphinxASR()
    else:
        raise Exception("Invalid ASR")

    if args.audio_in == "dummy":
        from audio_in.Dummy import DummyAudioIn

        audio_in = DummyAudioIn()
    elif args.audio_in == "sr":
        from audio_in.SR import SRAudioIn

        audio_in = SRAudioIn(
            device_index=args.sr_audio_in_device_index,
            sample_rate=args.sr_audio_in_sample_rate,
        )
    else:
        raise Exception("Invalid audio in")

    if args.entrainment_strategy == "neutral":
        from entrainment_strategy.Neutral import NeutralEntrainmentStrategy

        entrainment_strategy = NeutralEntrainmentStrategy()
    elif args.entrainment_strategy == "matching":
        from entrainment_strategy.Matching import MatchingEntrainmentStrategy

        entrainment_strategy = MatchingEntrainmentStrategy()
    elif args.entrainment_strategy == "neural":
        from entrainment_strategy.Neural import NeuralEntrainmentStrategy

        entrainment_strategy = NeuralEntrainmentStrategy(
            checkpoint_path=args.neural_entrainment_ckpt, max_turns=args.max_turns
        )
    else:
        raise Exception("Invalid entrainment strategy")

    feature_extractors = []

    if args.feature_extractor is not None:
        from feature_extractor.Dummy import DummyFeatureExtractor
        from feature_extractor.Praat import PraatFeatureExtractor
        from feature_extractor.Rate import RateFeatureExtractor

        for feature_extractor in args.feature_extractor:
            if feature_extractor == "dummy":
                feature_extractors.append(
                    DummyFeatureExtractor(
                        args.dummy_extractor_config, args.dummy_extractor_fuzz
                    )
                )
            elif feature_extractor == "praat":
                feature_extractors.append(PraatFeatureExtractor())
            elif feature_extractor == "rate":
                feature_extractors.append(RateFeatureExtractor())
            else:
                raise Exception("Invalid feature extractor")

    if args.response_generator == "dummy":
        from response_generator.Dummy import DummyResponseGenerator

        response_generator = DummyResponseGenerator(
            response=args.dummy_response_generator_response
        )
    elif args.response_generator == "eliza":
        from response_generator.Eliza import ElizaResponseGenerator

        response_generator = ElizaResponseGenerator()
    elif args.response_generator == "dialogpt":
        from response_generator.DialoGPT import DialoGPTResponseGenerator

        response_generator = DialoGPTResponseGenerator(size=args.dialogpt_size)
        pass
    else:
        raise Exception("Invalid response generator")

    if args.tts == "terminal":
        from tts.Dummy import DummyTTS

        tts = DummyTTS()
    elif args.tts == "tacotron2":
        from tts.Tacotron2 import Tacotron2TTS

        tts = Tacotron2TTS(
            checkpoint_path=args.tacotron2_ckpt,
            device_index=args.tacotron2_device_index,
        )
    elif args.tts == "pyttsx3":
        from tts.PyTTSx3 import PyTTSx3TTS

        tts = PyTTSx3TTS()
    elif args.tts == "dummy":
        from tts.Dummy import DummyTTS

        tts = DummyTTS()
    else:
        raise Exception("Invalid TTS")

    transformers = []
    entrainment_transformers = []

    if args.transformers is not None:
        from transformer.Log import LogTransformer
        from transformer.Map import MapTransformer
        from transformer.Normalize import NormalizeTransformer
        from transformer.NormalizeUser import NormalizeUserTransformer
        from transformer.Passthrough import PassthroughTransformer
        from transformer.PitchRange import PitchRangeTransformer

        for transformer in args.transformers:
            if transformer == "passthrough":
                transformers.append(PassthroughTransformer())
            elif transformer == "map":
                transformers.append(
                    MapTransformer(
                        args.map_transformer_target_config,
                        args.map_transformer_speaker_config,
                    )
                )
            elif transformer == "log":
                transformers.append(LogTransformer())
            elif transformer == "pitch_range":
                transformers.append(PitchRangeTransformer())
            elif transformer == "normalize":
                transformers.append(
                    NormalizeTransformer(config=args.normalize_transformer_config)
                )
            elif transformer == "normalize_user":
                transformers.append(NormalizeUserTransformer())
            else:
                raise Exception("Invalid transformer")

    if args.entrainment_transformers is not None:
        from entrainment_transformer.Map import (
            MapTransformer as EntrainmentMapTransformer,
        )
        from entrainment_transformer.Passthrough import (
            PassthroughTransformer as EntrainmentPassthroughTransformer,
        )
        from entrainment_transformer.Expand import (
            ExpandTransformer as EntrainmentExpandTransformer,
        )

        for transformer in args.entrainment_transformers:
            if transformer == "passthrough":
                entrainment_transformers.append(EntrainmentPassthroughTransformer())
            elif transformer == "map":
                entrainment_transformers.append(
                    EntrainmentMapTransformer(
                        args.map_transformer_target_config,
                        args.map_transformer_speaker_config,
                    )
                )
            elif transformer == "expand":
                entrainment_transformers.append(
                    EntrainmentExpandTransformer(args.expand_transformer_target_config)
                )
            else:
                raise Exception(f"Invalid transformer {transformer}")

    print("Starting...")

    context = Context()

    for i in range(args.max_turns):
        audio_in(context)
        asr(context)

        response_generator(context)

        for feature_extractor in feature_extractors:
            feature_extractor(context)

        context.begin_user_transform()

        for transformer in transformers:
            transformer(context)

        entrainment_strategy(context)

        for transformer in entrainment_transformers:
            transformer(context)

        tts(context)
