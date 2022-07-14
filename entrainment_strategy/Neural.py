import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
from torchtext.data import get_tokenizer
from torchtext.vocab import GloVe


class EmbeddingIterator:
    def __init__(self, embedding_seq, embedding_len, batch_size, device):
        self.embedding_seq = embedding_seq
        self.embedding_len = embedding_len
        self.batch_size = batch_size
        self.device = device

    def __iter__(self):
        self.next_embedding_start = torch.zeros(
            self.batch_size, dtype=torch.long, device=self.device
        )

        self.i = 0

        return self

    def __next__(self):
        next_embedding, next_embedding_len = self.peek()

        self.i += 1
        self.next_embedding_start += next_embedding_len

        return next_embedding, next_embedding_len

    def peek(self):
        next_embedding = []
        next_embedding_len = self.embedding_len[:, self.i]

        for batch_embeddings, start, length in zip(
            self.embedding_seq, self.next_embedding_start, next_embedding_len
        ):
            next_embedding.append(batch_embeddings[start : start + length])

        next_embedding = nn.utils.rnn.pad_sequence(
            next_embedding, batch_first=True, padding_value=1
        )

        return next_embedding, next_embedding_len


class Encoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers=2, dropout=0.0):
        super().__init__()

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.prenet = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU())

        self.rnn = nn.ModuleList(
            [nn.LSTMCell(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)

    def get_hidden(self, batch_size, device):
        return [
            (
                torch.zeros(batch_size, self.hidden_dim, device=device),
                torch.zeros(batch_size, self.hidden_dim, device=device),
            )
            for _ in range(self.num_layers)
        ]

    def forward(self, encoder_input, hidden):
        if len(hidden) != self.num_layers:
            raise Exception(
                "Number of hidden tensors must equal the number of RNN layers!"
            )

        x = self.prenet(encoder_input)

        hidden_out = []

        for h, rnn in zip(hidden, self.rnn):
            (h_out, c_out) = rnn(x, h)
            x = self.dropout(h_out)
            hidden_out.append((x, c_out))

        return x, hidden_out


class Attention(nn.Module):
    def __init__(self, history_in_dim, latest_in_dim, att_dim):
        super().__init__()

        self.att_w1 = nn.Linear(history_in_dim, att_dim, bias=False)
        self.att_w2 = nn.Linear(latest_in_dim, att_dim, bias=False)
        self.att_v = nn.Linear(att_dim, 1, bias=False)

    def forward(self, history, latest, mask):
        w1 = self.att_w1(history)
        w2 = self.att_w2(latest).unsqueeze(1)
        score = self.att_v(torch.tanh(w1 + w2))

        score = score.masked_fill(mask, float("-inf"))
        score = torch.softmax(score, dim=1)
        score_out = score.detach()
        score = score.squeeze(-1)

        score = score.unsqueeze(1)
        att_applied = torch.bmm(score, history)
        att_applied = att_applied.squeeze(1)

        return att_applied, score_out


class EmbeddingEncoder(nn.Module):
    def __init__(
        self,
        embedding_dim,
        embedding_encoder_out_dim,
        embedding_encoder_dropout,
        embedding_attention_dim,
    ):
        super().__init__()

        lstm_out_dim = embedding_encoder_out_dim // 2

        self.embedding_encoder = nn.LSTM(
            embedding_dim,
            lstm_out_dim,
            bidirectional=True,
            num_layers=2,
            dropout=embedding_encoder_dropout,
            batch_first=True,
        )

        self.embedding_attention = Attention(
            history_in_dim=embedding_encoder_out_dim,
            latest_in_dim=embedding_encoder_out_dim,
            att_dim=embedding_attention_dim,
        )

    def forward(self, embedding_input, embedding_len, device):
        batch_size = embedding_input.shape[0]

        embedding_encoder_input = nn.utils.rnn.pack_padded_sequence(
            embedding_input,
            embedding_len.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        embedding_encoder_output, (embedding_h, _) = self.embedding_encoder(
            embedding_encoder_input
        )

        # Dimensions: layers, directions, batch, hidden size
        embedding_h = embedding_h.view(2, 2, batch_size, -1)
        embedding_h = embedding_h[1].swapaxes(0, 1).reshape(batch_size, -1)

        embedding_encoder_output, _ = nn.utils.rnn.pad_packed_sequence(
            embedding_encoder_output, batch_first=True
        )

        embedding_mask = torch.arange(embedding_len.max(), device=device)
        embedding_mask = embedding_mask.unsqueeze(0).repeat(len(embedding_len), 1)
        embedding_mask = embedding_mask >= embedding_len.unsqueeze(1)
        embedding_mask = embedding_mask.unsqueeze(2)

        return self.embedding_attention(
            embedding_encoder_output, embedding_h, embedding_mask
        )


class Decoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers=2, dropout=0.0):
        super().__init__()

        self.num_layers = num_layers
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim

        self.rnn = nn.ModuleList([nn.LSTMCell(in_dim, hidden_dim)])
        for i in range(num_layers - 1):
            self.rnn.append(nn.LSTMCell(hidden_dim, hidden_dim))

        self.dropout = nn.Dropout(dropout)

    def get_hidden(self, batch_size, device):
        return [
            (
                torch.zeros(batch_size, self.hidden_dim, device=device),
                torch.zeros(batch_size, self.hidden_dim, device=device),
            )
            for _ in range(self.num_layers)
        ]

    def forward(self, decoder_input, hidden, at_idx=None):
        if len(hidden) != self.num_layers:
            raise Exception(
                "Number of hidden tensors must equal the number of RNN layers!"
            )

        hidden_out = []

        x = decoder_input

        for (h_in, c_in), rnn in zip(hidden, self.rnn):
            hidden = (
                (h_in[at_idx], c_in[at_idx]) if at_idx is not None else (h_in, c_in)
            )

            (h_out, c_out) = rnn(x, hidden)
            x = self.dropout(h_out)

            if at_idx is not None:
                h_in[at_idx] = h_out.type(h_in.dtype)
                h_out = h_in
                c_in[at_idx] = c_out.type(c_in.dtype)
                c_out = c_in

            hidden_out.append((h_out, c_out))

        return x, hidden_out


class EntrainmentModel(pl.LightningModule):
    modes = tuple(["us", "both"])

    def __init__(
        self,
        lr=0.0001,
        feature_dim=7,
        feature_encoder_out_dim=64,
        feature_encoder_dropout=0.5,
        feature_attention_dim=64,
        decoder_out_dim=64,
        decoder_dropout=0.5,
        training_mode="us",
        training_loss_mode="us",
        evaluation_mode="us",
        evaluation_loss_mode="us",
        teacher_forcing_mode="us",
        embeddings=False,
        embedding_dim=None,
        embedding_encoder_out_dim=None,
        embedding_encoder_dropout=None,
        embedding_attention_dim=None,
        teacher_forcing=0.5,
        lookahead_embedding_encoder=False,
        teacher_forcing_scheduler=None,
    ):
        super().__init__()

        self.lr = lr

        if training_mode not in EntrainmentModel.modes:
            raise Exception(
                f"Invalid training mode {training_mode}, legal values are {EntrainmentModel.modes}"
            )
        self.training_mode = training_mode

        if training_loss_mode not in EntrainmentModel.modes:
            raise Exception(
                f"Invalid training loss mode {training_loss_mode}, legal values are {EntrainmentModel.modes}"
            )
        self.training_loss_mode = training_loss_mode

        if evaluation_mode not in EntrainmentModel.modes:
            raise Exception(
                f"Invalid evaluation mode {evaluation_mode}, legal values are {EntrainmentModel.modes}"
            )
        self.evaluation_mode = evaluation_mode

        if evaluation_loss_mode not in EntrainmentModel.modes:
            raise Exception(
                f"Invalid evaluation loss mode {evaluation_loss_mode}, legal values are {EntrainmentModel.modes}"
            )
        self.evaluation_loss_mode = evaluation_loss_mode

        if teacher_forcing_mode not in EntrainmentModel.modes:
            raise Exception(
                f"Invalid teacher forcing mode {teacher_forcing_mode}, legal values are {EntrainmentModel.modes}"
            )
        self.teacher_forcing_mode = teacher_forcing_mode

        self.teacher_forcing = teacher_forcing
        self.teacher_forcing_scheduler = teacher_forcing_scheduler

        if self.teacher_forcing_scheduler is not None:
            iter(self.teacher_forcing_scheduler)

        self.feature_encoder_out_dim = feature_encoder_out_dim
        self.embedding_attention_dim = embedding_attention_dim

        self.feature_encoder = Encoder(
            in_dim=feature_dim + embedding_encoder_out_dim + 2,
            hidden_dim=feature_encoder_out_dim,
            num_layers=2,
            dropout=feature_encoder_dropout,
        )
        self.feature_attention = Attention(
            history_in_dim=feature_encoder_out_dim,
            latest_in_dim=feature_encoder_out_dim + embedding_encoder_out_dim,
            att_dim=feature_attention_dim,
        )

        self.embeddings = embeddings
        if embeddings:
            self.embedding_encoder = EmbeddingEncoder(
                embedding_dim,
                embedding_encoder_out_dim,
                embedding_encoder_dropout,
                embedding_attention_dim,
            )

            if lookahead_embedding_encoder:
                self.lookahead_embedding_encoder = EmbeddingEncoder(
                    embedding_dim,
                    embedding_encoder_out_dim,
                    embedding_encoder_dropout,
                    embedding_attention_dim,
                )

            self.has_lookahead_embedding_encoder = lookahead_embedding_encoder

        self.decoder = Decoder(
            feature_encoder_out_dim + embedding_encoder_out_dim,
            decoder_out_dim,
            dropout=decoder_dropout,
        )

        self.linear = nn.Sequential(
            nn.Linear(decoder_out_dim, decoder_out_dim),
            nn.ReLU(),
            nn.Linear(decoder_out_dim, feature_dim),
        )

    def forward(
        self,
        i,
        feature_input,
        feature_history,
        feature_mask,
        feature_encoder_hidden,
        decoder_hidden,
        speaker,
        embedding_input=None,
        embedding_len=None,
        pred_embedding_input=None,
        pred_embedding_len=None,
        pred_idxs=None,
    ):
        if self.embeddings:
            if embedding_input is None or embedding_len is None:
                raise Exception("Model requires embedding input but data was missing!")

        batch_size = feature_input.shape[0]

        att_scores = {}

        # If a set of prediction batch indices was not given, then assume we are predicting
        # for the entire batch
        if pred_idxs is None:
            pred_idxs = torch.arange(batch_size)

        output = torch.zeros_like(feature_input)

        # Step 1: Encoder
        # ------------------------------------------------------------------------
        feature_encoder_input = [feature_input]
        feature_encoder_input.append(speaker)

        if self.embeddings:
            # If we are doing word embeddings, we need to encode them first.
            # This step produces a turn-level embedding from the transcript.
            embedding_output, embedding_scores = self.embedding_encoder(
                embedding_input, embedding_len, self.device
            )
            att_scores["embedding"] = embedding_scores

            # If we are looking ahead to word embeddings from the turn we are currently
            # predicting, compute those here
            lookahead_embedding_encoder = (
                self.lookahead_embedding_encoder
                if self.has_lookahead_embedding_encoder
                else self.embedding_encoder
            )
            if len(pred_embedding_input) > 0:
                (
                    pred_embedding_output,
                    pred_embedding_scores,
                ) = lookahead_embedding_encoder(
                    pred_embedding_input, pred_embedding_len, self.device
                )
                att_scores["pred_embedding"] = pred_embedding_scores

            # We treat the turn-level embedding as another feature, so we add
            # it to the feature encoder input.
            feature_encoder_input.append(embedding_output)

        feature_encoder_input = torch.cat(feature_encoder_input, dim=-1)

        feature_encoder_output, feature_encoder_hidden = self.feature_encoder(
            feature_encoder_input, feature_encoder_hidden
        )

        feature_history[:, i] = feature_encoder_output

        if len(pred_idxs) == 0:
            att_scores["feature"] = torch.zeros((batch_size, i + 1, 1))
            return (
                output,
                att_scores,
                feature_encoder_hidden,
                decoder_hidden,
            )

        # Step 2: Turn-level attention
        # ------------------------------------------------------------------------

        # We only want to compute the attention for batches where we are predicting this timestep.
        feature_attention_input = [feature_encoder_output[pred_idxs]]

        # Optionally, concatenate the encoded feature output with the upcoming predicted
        # turn's word embeddings.
        if self.embeddings:
            feature_attention_input.append(pred_embedding_output)

        feature_attention_input = torch.cat(feature_attention_input, dim=-1)

        feature_encoded, feature_scores = self.feature_attention(
            feature_history[pred_idxs, : i + 1],
            feature_attention_input,
            feature_mask[pred_idxs, : i + 1],
        )

        feature_scores_expanded = torch.zeros(
            (batch_size, i + 1, 1), device=self.device
        )
        feature_scores_expanded[pred_idxs] = feature_scores
        att_scores["feature"] = feature_scores_expanded

        # Step 2: Decoder
        # ------------------------------------------------------------------------
        decoder_input = [feature_encoded]

        # Optionally, concatenate the decoder input with the upcoming predicted
        # turn's word embeddings.
        if self.embeddings:
            decoder_input.append(pred_embedding_output)

        decoder_input = torch.cat(decoder_input, dim=-1)

        decoder_output, decoder_hidden = self.decoder(
            decoder_input, decoder_hidden, pred_idxs
        )

        output_features = self.linear(decoder_output)

        output = torch.zeros_like(feature_input, dtype=output_features.dtype)
        output[pred_idxs] = output_features

        return output, att_scores, feature_encoder_hidden, decoder_hidden


class NeuralEntrainmentStrategy:
    def __init__(self, checkpoint_path, max_turns):
        self.model = EntrainmentModel.load_from_checkpoint(
            checkpoint_path,
            feature_dim=7,
            feature_encoder_out_dim=256,
            feature_encoder_dropout=0.5,
            feature_attention_dim=128,
            decoder_out_dim=256,
            evaluation_mode="us",
            evaluation_loss_mode="us",
            training_mode="us",
            training_loss_mode="us",
            teacher_forcing_mode="us",
            embeddings=True,
            embedding_dim=300,
            embedding_encoder_out_dim=256,
            embedding_encoder_dropout=0.5,
            embedding_attention_dim=128,
            teacher_forcing=0.5,
            lookahead_embedding_encoder=False,
        ).cuda()

        self.max_turns = max_turns * 2
        self.turn = 0

        self.feature_encoder_hidden = self.model.feature_encoder.get_hidden(
            batch_size=1, device="cuda"
        )

        self.decoder_hidden = self.model.decoder.get_hidden(batch_size=1, device="cuda")

        self.feature_history = torch.zeros((1, self.max_turns, 256)).cuda()

        self.speaker_history = torch.zeros((1, self.max_turns, 2)).cuda()

        self.features = [
            "pitch_mean",
            "pitch_range",
            "intensity_mean",
            "jitter",
            "shimmer",
            "nhr",
            "rate",
        ]

        self.tokenizer = get_tokenizer("basic_english")
        self.vec = GloVe(name="840B", dim=300)

    def __call__(self, context):
        self.model.eval()

        our_text = (
            self.vec.get_vecs_by_tokens(
                self.tokenizer(context.get_latest_response_text())
            )
            .unsqueeze(0)
            .cuda()
        )
        our_text_len = torch.tensor([our_text.shape[1]]).cuda()

        user_features = (
            torch.tensor(
                [
                    context.get_transformed_user_feature_values(f)[-1]
                    for f in self.features
                ],
                dtype=torch.float,
            )
            .unsqueeze(0)
            .cuda()
        )

        user_text = (
            self.vec.get_vecs_by_tokens(self.tokenizer(context.get_latest_user_text()))
            .unsqueeze(0)
            .cuda()
        )
        user_text_len = torch.tensor([user_text.shape[1]]).cuda()

        with torch.no_grad():
            # Once to compute our output
            (
                entrained_features,
                scores,
                self.feature_encoder_hidden,
                self.decoder_hidden,
            ) = self.model(
                i=self.turn,
                feature_input=user_features,
                feature_history=self.feature_history,
                feature_mask=torch.zeros(
                    (1, self.max_turns, 1), dtype=torch.bool
                ).cuda(),
                feature_encoder_hidden=self.feature_encoder_hidden,
                decoder_hidden=self.decoder_hidden,
                pred_idxs=torch.tensor([0]).cuda(),
                speaker=torch.tensor([[1.0, 0.0]]).cuda(),
                embedding_input=user_text,
                embedding_len=user_text_len,
                pred_embedding_input=our_text,
                pred_embedding_len=our_text_len,
            )

            self.turn += 1

            # Again to incorporate our output into the dialogue history
            (_, _, self.feature_encoder_hidden, self.decoder_hidden,) = self.model(
                i=self.turn,
                feature_input=entrained_features,
                feature_history=self.feature_history,
                feature_mask=torch.zeros(
                    (1, self.max_turns, 1), dtype=torch.bool
                ).cuda(),
                feature_encoder_hidden=self.feature_encoder_hidden,
                decoder_hidden=self.decoder_hidden,
                pred_idxs=torch.tensor([]).cuda(),
                speaker=torch.tensor([[0.0, 1.0]]).cuda(),
                embedding_input=our_text,
                embedding_len=our_text_len,
                pred_embedding_input=torch.tensor([]).cuda(),
                pred_embedding_len=torch.tensor([]).cuda(),
            )

            self.turn += 1

        entrained_features = entrained_features.squeeze(0)

        for i, f in enumerate(self.features):
            context.set_partner_entrained_feature_value(f, entrained_features[i])
