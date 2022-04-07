import tempfile
import re
import math
import parselmouth

MIN_DUR = 6.4 / 75


def get_jitter_and_shimmer(sound, pitch, pulses):
    if pitch is None or pulses is None:
        return None, None

    mean_pitch = parselmouth.praat.call(pitch, "Get mean", 0, 0, "Hertz")
    mean_period = 1.0 / mean_pitch
    num_voiced_frames = parselmouth.praat.call(pitch, "Count voiced frames")

    if num_voiced_frames <= 0:
        return None, None

    textgrid = parselmouth.praat.call(pulses, "To TextGrid (vuv)", 0.02, mean_period)
    intervals = parselmouth.praat.call(
        [sound, textgrid], "Extract intervals", 1, "no", "V"
    )

    if type(intervals) is not list:
        intervals = [intervals]

    concatted = parselmouth.Sound.concatenate(intervals)

    if concatted.get_total_duration() <= MIN_DUR:
        return None, None

    concatted_pitch = concatted.to_pitch()
    concatted_pulses = parselmouth.praat.call([concatted_pitch], "To PointProcess")

    jitter = parselmouth.praat.call(
        concatted_pulses, "Get jitter (local)", 0.0, 0.0, 0.0001, 0.02, 1.3
    )

    shimmer = parselmouth.praat.call(
        [concatted, concatted_pulses],
        "Get shimmer (local)",
        0.0,
        0.0,
        0.0001,
        0.02,
        1.3,
        1.6,
    )

    return jitter, shimmer


nhr_re = re.compile("Mean noise-to-harmonics ratio: (\d*\.?\d+)")
jitter_re = re.compile("Jitter \(local\): (\d*\.?\d+)%")
shimmer_re = re.compile("Shimmer \(local\): (\d*\.?\d+)%")


def get_features(sound):
    try:
        pitch = sound.to_pitch()
        pulses = parselmouth.praat.call([sound, pitch], "To PointProcess (cc)")
    except:
        pitch = None
        pulses = None

    jitter, shimmer = get_jitter_and_shimmer(sound, pitch, pulses)

    intensity_mean = intensity_std = intensity_min = intensity_max = energy = None
    if sound.get_total_duration() > MIN_DUR:
        intensity = sound.to_intensity()

        intensity_mean = parselmouth.praat.call(
            intensity, "Get mean", 0.0, 0.0, "Energy"
        )
        intensity_max = parselmouth.praat.call(
            intensity, "Get maximum", 0.0, 0.0, "Parabolic"
        )

        intensity_min = parselmouth.praat.call(
            intensity, "Get minimum", 0.0, 0.0, "Parabolic"
        )

    nhr = None

    if sound and pitch and pulses:
        report = str(
            parselmouth.praat.call(
                [sound, pitch, pulses],
                "Voice report",
                0,
                0,
                75,
                500,
                1.3,
                1.6,
                0.03,
                0.45,
            )
        )

        nhr = nhr_re.search(report)

        if nhr:
            nhr = nhr.group(1)

    pitch_mean, pitch_max, pitch_min, pitch_range = None, None, None, None

    if pitch is not None:
        pitch_mean = (
            parselmouth.praat.call(pitch, "Get mean", 0.0, 0.0, "logHertz")
            if pitch
            else None
        )

        pitch_95 = parselmouth.praat.call(
            pitch, "Get quantile", 0.0, 0.0, 0.95, "logHertz"
        )
        pitch_05 = parselmouth.praat.call(
            pitch, "Get quantile", 0.0, 0.0, 0.05, "logHertz"
        )
        pitch_range = pitch_95 - pitch_05

    return {
        "pitch_mean": pitch_mean,
        "pitch_range": pitch_range,
        "intensity_mean": math.log(intensity_mean),
        "jitter": math.log(jitter),
        "shimmer": math.log(shimmer),
        "nhr": math.log(float(nhr)),
        "duration": sound.duration,
    }


class PraatFeatureExtractor:
    def __call__(self, context):
        audio = context.get_latest_user_audio()

        wav_data = audio.get_wav_data()

        with tempfile.NamedTemporaryFile() as tmp:
            tmp.write(wav_data)
            sound = parselmouth.Sound(tmp.name)

        for feature, value in get_features(sound).items():
            context.add_feature_value(feature, value)
