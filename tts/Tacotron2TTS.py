import sys

sys.path.append("../tacotron2")

import librosa
import numpy as np
import pytorch_lightning as pl
import sounddevice
import torch
from model.tacotron2 import Tacotron2
from sklearn.preprocessing import OrdinalEncoder

ALLOWED_CHARS = "!'(),.:;? \\-ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

FEATURES = [
    "pitch_mean",
    "pitch_range",
    "intensity_mean",
    "jitter",
    "shimmer",
    "nhr",
    "duration",
]


class Tacotron2TTS:
    def __init__(self):
        self.tacotron2 = Tacotron2.load_from_checkpoint(
            checkpoint_path="tts/tacotron2/tacotron-features-all-v2-22040.ckpt",
            lr=1e-4,
            weight_decay=0,
            num_chars=len(ALLOWED_CHARS) + 1,
            encoder_kernel_size=5,
            num_mels=80,
            char_embedding_dim=512,
            prenet_dim=256,
            att_rnn_dim=1024,
            att_dim=128,
            rnn_hidden_dim=1024,
            postnet_dim=512,
            dropout=0.5,
            teacher_forcing=False,
            speech_feature_dim=7,
            speech_features=True,
        ).cuda()

        self.end_token = "^"
        self.encoder = OrdinalEncoder()
        self.encoder.fit([[x] for x in list(ALLOWED_CHARS) + [self.end_token]])

    def _translate(self, value, leftMin, leftMax, rightMin, rightMax):
        # Figure out how 'wide' each range is
        leftSpan = leftMax - leftMin
        rightSpan = rightMax - rightMin

        # Convert the left range into a 0-1 range (float)
        valueScaled = float(value - leftMin) / float(leftSpan)

        # Convert the 0-1 range into a value in the right range.
        return rightMin + (valueScaled * rightSpan)

    def __call__(self, context):
        text = context.get_latest_response_text()

        feature_vector = []
        for feat_name in FEATURES:
            f = np.array(context.get_feature_values(feat_name))
            print(feat_name, f)

            F_std = f.std()
            F_med = np.median(f)
            F_min = F_med - (3 * F_std)
            F_max = F_med + (3 * F_std)

            if F_std == 0:
                f_val = 0.
            else:
                f_val = self._translate(f[-1], F_min, F_max, -1, 1)
            
            feature_vector.append(f_val)

        print("Generating response...")
        print("Using feature vector " + str(feature_vector))

        encoded = (
            torch.LongTensor(
                self.encoder.transform([[x] for x in text.lower()] + [[self.end_token]])
            )
            .squeeze(1)
            .unsqueeze(0)
        ) + 1

        tts_data = {
            "chars_idx": torch.LongTensor(encoded).cuda(),
            "mel_spectrogram": torch.zeros((1, 600, 80)).cuda(),
        }
        tts_metadata = {
            "chars_idx_len": torch.IntTensor([encoded.shape[1]]).cuda(),
            "mel_spectrogram_len": torch.IntTensor([600]).cuda(),
            "features": torch.Tensor([feature_vector]).cuda(),
        }

        with torch.no_grad():
            self.tacotron2.eval()
            mels, mels_post, gates, alignments = self.tacotron2.predict_step(
                (tts_data, tts_metadata), 0, 0
            )

        mels = mels.cpu()
        mels_post = mels_post.cpu()
        gates = gates.cpu()
        alignments = alignments.cpu()

        gates = gates[0]
        gates = torch.sigmoid(gates)
        end = -1
        for i in range(gates.shape[0]):
            if gates[i][0] < 0.5:
                end = i
                break

        del mels
        del gates
        del alignments

        mels_exp = torch.exp(mels_post[0])[:end]
        wav = librosa.feature.inverse.mel_to_audio(
            mels_exp.numpy().T,
            sr=22050,
            n_fft=1024,
            win_length=1024,
            hop_length=256,
            power=1,
        )

        print("Playing response...")

        sounddevice.play(wav, samplerate=22050, device=26, blocking=True)
