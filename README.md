# Brooklyn Speech Lab Dialogue Engine

This repository contains a modular spoken dialogue system capable of engaging a user in conversation. The system is designed so that many different audio recording, speech recognition, conversation model, feature transformation, and TTS modules can be swapped out to change its behavior.

The system was designed primarily for [entrainment research](https://academiccommons.columbia.edu/doi/10.7916/D8XP7DBD/download), so a key component of the system is swappable entrainment strategies. An entrainment strategy is a specialized component capable of directing the output of a TTS module, changing any available acoustic or prosodic parameters in response to those of its human conversation partner. As an example, a simple entrainment strategy included in the repository will make the TTS approximately match its partner's average vocal pitch and speaking rate over their turn in the conversation.

While the dialogue engine is under active development, it is currently limited in that it offers mostly simple modules. However, two advanced modules are available: a custom implementation of [Tacotron 2](https://arxiv.org/pdf/1712.05884.pdf) for TTS, adapted from [my Tacotron repository](https://github.com/mattm458/tacotron2) and trained with several acoustic-prosodic controls; and a [DialoGPT](https://github.com/microsoft/DialoGPT) Hugging Face transformer for dialogue response generation.

The following additional modules are under active research, in order from most to least activity:

* Neural entrainment strategies for more realistic entrainment behavior, learned from real-world conversation corpora like [Fisher](http://www.lrec-conf.org/proceedings/lrec2004/pdf/767.pdf), [Switchboard](https://www.computer.org/csdl/proceedings-article/icassp/1992/00225858/12OmNxGSmbC), the [Columbia Games Corpus](https://academiccommons.columbia.edu/doi/10.7916/D8BK1MMW/download), and our in-house Brooklyn Multi-Interaction Corpus (to be presented at LREC 2022).
* TTS engines (specifically Tacotron) with controllable acoustic and prosodic parameters. We are interested in determining which parameters can be controlled reliably, as well as determining whether the model meets the targets we give it.
* Fast neural vocoders, such as [WaveRNN](https://proceedings.mlr.press/v80/kalchbrenner18a/kalchbrenner18a.pdf), and whether they can contribute to producing output with controllable acoustic and prosodic parameters.

Currently, the engine is a proof of concept and unsuitable for use in a production environment. It is evolving in response to our research, and our ultimate goal is to use it to evaluate different entrainment strategies in real human-computer conversations. Consequently, we expect many aspects of its architecture to change over time.

## Installation

The dialogue engine was developed with Python 3.9 on a Linux machine, and has not been tested on anything else. It is likely compatible with Python 3.6+, but it is not compatible with Python 3.10 due to its dependencies.

The included `requirements.txt` file should be sufficient for most uses, though there are some caveats to consider.


### PyTorch
This project has a dependency on [PyTorch](https://pytorch.org/get-started/locally/) for the Tacotron TTS. The `requirements.txt` file is configured to install the current stable release of PyTorch with CUDA 11.3. If you need a different version of CUDA, or you intend to run the engine on a CPU (not recommended) or to use PyTorch's beta ROCm support on AMD GPUs, you will need to alter `requirements.txt`. Mac users are on their own, since there appears to be some complexity around installing PyTorch with CUDA support on this platform that I am not familiar with.

### pyttsx3
This project has a dependency on [pyttsx3](https://github.com/nateshmbhat/pyttsx3) for non-Tacotron TTS. If you run Linux, please check the directions linked above - you must manually install either the espeak or espeak-ng TTS engine. I am not sure what happens on Mac or Windows so you will need to play around with this. However, if necessary, [espeak is available in Homebrew](https://formulae.brew.sh/formula/espeak) for Mac users if necessary.

Please note that requirements.txt installs a version of pyttsx3 from GitHub instead of from the official PyPI repositories. This is because the latest pyttsx3 code in GitHub supports both pitch manipulation and espeak-ng, but it has not yet been packaged for a formal release.

### System requirements
As long as you do not use the Tacotron TTS module or the DialoGPT dialogue module, the engine should run on most systems without specialized hardware. Using either DialoGPT or Tacotron requires a GPU, and using both together likely requires a powerful GPU with lots of memory.

## Usage

To run the dialogue engine, run

```
python main.py
```


### Audio setup

If you use the Tacotron TTS module or the SpeechRecognition ASR module (described in the "Module Reference" section below), you must specify the index of an appropriate audio device for each of the modules to use. Specifically, you must have a speaker for Tacotron and a microphone for SpeechRecognition.

To determine the index of audio devices available on your computer, you can use the sounddevice Python module (it is installed as a dependency):

```
python -m sounddevice
```

This will list your computer's audio devices and their indices. Once you find the input and output devices you wish to use, you can give their indices as arguments to the appropriate module.

### CLI options

#### Engine configuration

By default, the engine starts in a minimalist debug mode that does not produce audio output or record audio, and only communicates via the command line. You can expand the functionality of the dialogue engine by supplying the following optional command line arguments:

|Argument|Description|
|--------|-----------|
|<nobr>`--max-turns`</nobr>|Configure a maximum number of turns for the conversation, after which the engine will stop and print a summary. Defaults to 5.|
|<nobr>`--asr`</nobr>|Specify an ASR module. Valid values include `terminal` and `sphinx`. Defaults to `terminal`.|
|<nobr>`--audio-in`</nobr>|Specify an audio in module. Valid values include `dummy` and `sr`. Defaults to `dummy`.|
|<nobr>`--entrainment-strategy`</nobr>|Specify an entrainment strategy module. Valid values include `neutral` and `matching`. Defaults to `neutral`.|
|<nobr>`--feature-extractor`</nobr>|Specify a comma-separated list of feature extractors in the order they should be run in. Available values include `dummy`, `praat` and `rate`. Defaults to none.|
|<nobr>`--response-generator`</nobr>|Specify a response generator module. Valid values include `dummy`, `eliza`, and `dialogpt`. Defaults to `eliza`.|
|<nobr>`--transformers`</nobr>|Specify an output value transformer. Valid values include `passthrough`, `map`, `pitch_range`, `log`, and `normalize`.|
|<nobr>`--tts`</nobr>|Specify a TTS module. Valid values include `terminal`, `pyttsx3`, and `tacotron2`. Defaults to `terminal`.

Depending on the modules you select, you may need to specify additional command line parameters. These are detailed in the "Module Reference" section below.

## Module reference

### ASR

ASR modules are responsible for translating a given speech recording into a text transcription.

|Module|Description|Parameters|
|------|-----------|-------|
|Terminal|Does not perform speech recognition. Instead, prompts the user for textual input at the command line. Useful for development and debugging.|None|
|Sphinx|Uses the Sphinx offline ASR system for recognizing speech.|None|

### Audio In

Audio in modules are responsible for recording the computer's human partner. When invoked, they pause execution until a complete utterance has been recorded.

|Module|Description|Parameters|
|------|-----------|-------|
|Dummy|A no-op module that does not record audio. Ideally used with the TerminalASR module described below.|None|
|SR|Uses the SpeechRecognition audio in module to record audio. The module takes advantage of SpeechRecognition features, including producing recordings suitable for feeding into the SphinxASR module, and detecting and minimizing background noise.|<nobr>`--sr-audio-in-device-index`</nobr>: The index of the audio device to use for recording. **Required**.<br/><br/><nobr>`--sr-audio-in-sample-rate`</nobr>: The sample rate to record in hertz. Defaults to 32000.

### Entrainment Strategy

An entrainment strategy is responsible for determining appropriate feature values for the TTS module's response given any available information in the context object. This can include extracted features, audio, and text, both from their human partner and from TTS output. They can draw from the latest turn or from the entire conversational history.

|Module|Description|Parameters|
|------|-----------|-------|
|Neutral|Does not entrain. Does not pass values through to the TTS, with the expectation that the TTS will choose appropriate neutral values.|None|
|Matching|Entrains by passing extracted features directly to the TTS without modification.|None|

### Feature Extractor

Feature extractors are responsible for producing simple acoustic and prosodic features from a recorded audio sample. A pipeline of several feature extractors can be constructed.

|Module|Description|Parameters|
|------|-----------|-------|
|Dummy|Does not extract features. Instead, passes predefined hardcoded feature values as defined in a JSON configuration file. Useful for development or debugging.|<nobr>`--dummy-extractor-config`</nobr>: Path to a JSON file containing hardcoded dummy entrainment values. A sample JSON file can be found in the `sample_config/` directory. **Required**.<br/><br/><nobr>`--dummy-extractor-fuzz`</nobr>: Randomly fuzz the values to a percentage within the given range. Defaults to 0.|
|Praat|Uses Praat (with Python bindings via [ParselMouth](https://github.com/YannickJadoul/Parselmouth)) to extract a set of acoustic-prosodic features commonly used by our lab for entrainment research: mean pitch, pitch range, mean intensity, jitter, shimmer, noise-to-harmonics ratio (NHR), and duration.|None|
|Rate|Uses a combination of NLTK syllable counts (for words in the NLTK dictionary), the Python Hyphenate package (for words not in the NLTK dictionary), and a variety of custom rules (imported from other projects, not always applicable to ASR output) to determine the rate of speech for an audio sample. **It requires a feature extractor capable of extracting the audio duration to have been run before it, since computing the rate requires duration**.|None|

### Response Generator

Response generators act as a chatbot, and must synthesize a textual response to the latest human partner's utterance.

|Module|Description|Parameters|
|------|-----------|-------|
|Dummy|Does not contain a dialogue model. Instead, it always produces the same response. Useful for development and debugging.|<nobr>`--dummy-response-generator-response`</nobr>: A string to respond with. Defaults to "Testing".|
|Eliza|Generates responses using the classic ELIZA parody therapist engine. To implement this, code from [this project](https://github.com/wadetb/eliza) was ported over.|None|
|DialoGPT|Generates responses using Microsoft's pretrained [DialoGPT](https://github.com/microsoft/DialoGPT) dialogue model. Requires a GPU for reasonable performance.|<nobr>`--dialogpt-size`</nobr>: The DialoGPT model size to download and use. Available options are `small`, `medium`, and `large`. Defaults to `medium`.

### TTS

TTS modules are responsible for synthesizing a spoken response from the text determined by the response generator.

|Module|Description|Parameters|
|------|-----------|-------|
|Dummy|Does nothing. Useful for development and debugging.|None|
|PyTTSx3|Uses the pyttsx3 engine to generate speech.|None|
|Tacotron2|Uses a Tacotron 2 model to generate speech. The Tacotron model produces Mel spectrogram output, which is converted to a waveform using librosa's Griffin-Lim implementation.|<nobr>`--tacotron2-ckpt`</nobr>: The path to a trained Tacotron model checkpoint. **Required**.<br/><br/><nobr>`--tacotron2-device-index`</nobr>: The index of the audio device to use for playback. **Required**.|


A Tacotron 2 model checkpoint is available [here](https://drive.google.com/file/d/1M2tzu6z8UpIabNdQOz5LsGI-VOsWifBH/view?usp=sharing).

### Transformers

Transformers are an intermediary between the entrainment strategy and TTS, performing necessary transformations to fit entrained feature values into a range appropriate for the TTS. A pipeline of several transformers can be constructed.

|Module|Description|Parameters|
|------|-----------|-------|
|Passthrough|Does nothing, only passing through raw entrained feature values as they are.|None|
|Map|Dynamically maps the entrained features from one vocal range to another. Useful in situations where the entrained feature values were naively chosen based on the speaker's vocal range, when in reality the TTS engine was trained on a different range. Uses the conversation history to determine typical values for the speaker, which can cause instability early in the conversation until a sufficient number of turns are recorded. Can be mitigated by supplying default fallback values for the speaker.|<nobr>`--map-transformer-target-config`:</nobr> The path to a JSON file containing means and standard deviations for each feature in the TTS's voice. **Required**.<br/><br/><nobr>`--map-transformer-speaker-config`:</nobr> The path to a JSON file containing means and standard deviations for each feature in the speaker's voice, or an approximation of the speaker's voice.|
|Normalize|Performs z-score normalization on feature values. Must be configured with a JSON file containing mean and standard deviations of each feature value, typically those related to the TTS's voice.|<nobr>`--normalize-transformer-config`</nobr>: The path to a JSON file containing means and standard deviations for each feature in the TTS's voice. **Required**.|
|Log|Takes the log of each feature value.|None|
|Pitch Range|Computes the pitch range from pitch_95 and pitch_05.|None|

## Typical Usage

### pyttsx3

### Tacotron 2 v2

This version of Tacotron requires the checkpoint found [here](https://drive.google.com/file/d/1M2tzu6z8UpIabNdQOz5LsGI-VOsWifBH/view?usp=sharing).

Depending on the entrainment strategy you use, you will probably need to set up the following set of transformers in this order:

1. Map the entrained feature values into the vocal range of the [LJSpeech dataset](https://keithito.com/LJ-Speech-Dataset/), which can be found in the `sample_config` directory as `tacotron2-ljspeech-target.json`.
2. Use the Pitch Range transformer to construct the `pitch_range` entrained feature value.
3. Take the log of each entrained feature.
4. Z-score normalize the entrained feature values.


If you use the Dummy entrainment strategy, it is not necessary to use any transformers unless the hardcoded values are not already in the appropriate range for Tacotron. There are three JSON files for different situations:

* `dummy_entrainment_strategy_tts.json`, which is hardcoded to values already in the appropriate range for the Tacotron model. It does not need to be transformed.
* `dummy_entrainment_strategy_ljspeech_range.json`, which is already in the LJSpeech vocal range. It must be transformed using the Pitch Range, Log, and Z-score transformers, but it does not need to be mapped.
* `dummy_entrainment_strategy_other_range.json`, which is in a constructed vocal range different from LJSpeech. It must be transformed using the complete 4-step pipeline shown above.

If you use the Neutral entrainment strategy, you do not need to use any transformers. 

We may create entrainment strategies in the future that automatically entrain into the correct range for a given TTS model. Transformers will not be necessary in these cases.

### Example Commands

Here are some example commands for the given scenarios above:

#### Without speech input

**Dummy feature extractor with values in a constructed vocal range. The system does not entrain, and always speaks in a neutral tone.**

```
python main.py \
	--tts tacotron2 \
	--tacotron2-ckpt tacotron-features-all-v2-22040.ckpt \
	--tacotron2-device-index <input device index> \
	--feature-extractor dummy \
	--dummy-extractor-config sample_config/dummy_feature_example.json \
	--dummy-extractor-fuzz 0.2
```
Alternate version with pyttsx3 for low-resource systems:
```
python main.py \
	--tts pyttsx3 \
	--feature-extractor dummy \
	--dummy-extractor-config sample_config/dummy_feature_example.json \
	--dummy-extractor-fuzz 0.2
```

**Dummy feature extractor with values in a constructed vocal range. The system attempts to entrain by matching the dummy values, which must be mapped to the LJSpeech vocal range and transformed into the Tacotron input range:**

```
python main.py \
	--tts tacotron2 \
	--tacotron2-ckpt tacotron-features-all-v2-22040.ckpt \
	--tacotron2-device-index <output device index> \
	--entrainment-strategy matching \
	--feature-extractor dummy \
	--dummy-extractor-config sample_config/dummy_feature_example.json \
	--dummy-extractor-fuzz 0.2 \
	--transformers map log pitch_range normalize \
	--map-transformer-target-config sample_config/ljspeech-range.json \
	--map-transformer-speaker-config sample_config/constructed-range.json \
	--normalize-transformer-config sample_config/ljspeech-log-range.json
```
Alternate version with pyttsx3 for low-resource systems:
```
python main.py \
	--tts pyttsx3 \
	--entrainment-strategy matching \
	--feature-extractor dummy \
	--dummy-extractor-config sample_config/dummy_feature_example.json \
	--dummy-extractor-fuzz 0.2
```

#### With speech input

**Microphone input used with the neutral entrainment strategy. The TTS will respond in a neutral tone, and will be unresponsive to your speech characteristics.**

```
python main.py \
	--audio-in sr \
	--sr-audio-in-device-index <input device index> \
	--asr sphinx \
	--tts tacotron2 \
	--tacotron2-ckpt tacotron-features-all-v2-22040.ckpt \
	--tacotron2-device-index <output device index>
```

Alternate version with pyttsx3 for low-resource systems: 
```
python main.py \
	--audio-in sr \
	--sr-audio-in-device-index <input device index> \
	--asr sphinx \
	--tts pyttsx3
```


**Microphone input with the usual feature extraction suite, with a matching entrainment strategy. The mapping configuration is set up without a fallback speaker configuration, so Tacotron will respond in a neutral tone until the mapper accumulates enough samples to transform your vocal range into LJSpeech's vocal range.
**
```
python main.py \
	--audio-in sr \
	--sr-audio-in-device-index <input device index> \
	--asr sphinx \
	--tts tacotron2 \
	--tacotron2-ckpt tacotron-features-all-v2-22040.ckpt \
	--tacotron2-device-index <output device index> \
	--entrainment-strategy matching \
	--feature-extractor dummy \
	--dummy-extractor-config sample_config/dummy_feature_example.json \
	--dummy-extractor-fuzz 0.2 \
	--transformers map log pitch_range normalize \
	--map-transformer-target-config sample_config/ljspeech-range.json \
	--map-transformer-speaker-config sample_config/constructed-range.json \
	--normalize-transformer-config sample_config/ljspeech-log-range.json
```
Alternate version with pyttsx3 for low-resource systems:

```
python main.py \
	--audio-in sr \
	--sr-audio-in-device-index <input device index> \
	--asr sphinx \
	--tts pyttsx3 \
	--entrainment-strategy matching \
	--feature-extractor dummy \
	--dummy-extractor-config sample_config/dummy_feature_example.json \
	--dummy-extractor-fuzz 0.2
```