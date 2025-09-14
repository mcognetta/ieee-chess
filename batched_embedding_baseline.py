import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math

from common import CLEAN_MEAN, CLEAN_STD

from scheduler import InverseSquareRootSchedule

import data_loader
from data_loader import MOVE_BUCKETS
import leela_utils
import maia_utils

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

TRAIN_BATCH_SIZE = 1024

(
    DIM,
    MOVE_EMBED_DIM,
    MAIA2_FEATURE_DIM,
    MAIA2_FEATURE_EMBEDDING_DIM,
    MAIA2_FEATURE_INTERMEDIATE_EMBEDDING_DIM,
) = (2048, 64, 22, 128, 512)


class BatchedBaseline(torch.nn.Module):
    def __init__(self, leela_model_types=[leela_utils.LEELA_TYPE.SMALL]):
        super(BatchedBaseline, self).__init__()
        self._leela_model_types = leela_model_types

        self._board_embedders = []
        self._board_embedding_dims = []
        self._board_embedding_dim_reducers = []
        for leela_model_type in self._leela_model_types:
            # This should handle both Leela and Maia models. Not sure if this is the "most optimal" way to do this.
            if isinstance(leela_model_type, leela_utils.LEELA_TYPE):
                self._board_embedders.append(
                    leela_utils.LeelaEmbedder(leela_model_type)
                )
            elif isinstance(leela_model_type, maia_utils.MAIA_TYPE):
                self._board_embedders.append(maia_utils.MaiaEmbedder(leela_model_type))
            else:
                raise ValueError(f"invalid type {leela_model_type}")
            self._board_embedding_dims.append(self._board_embedders[-1].embed_size())
            self._board_embedding_dim_reducers.append(
                torch.nn.Linear(self._board_embedders[-1].embed_size(), DIM)
            )
        self._board_embedders = torch.nn.ParameterList(self._board_embedders)
        self._board_embedding_dim_reducers = torch.nn.ParameterList(
            self._board_embedding_dim_reducers
        )

        self._num_move_embed = torch.nn.Embedding(len(MOVE_BUCKETS), MOVE_EMBED_DIM)
        self._maia2_feature_embedding_1 = torch.nn.Linear(
            MAIA2_FEATURE_DIM, MAIA2_FEATURE_INTERMEDIATE_EMBEDDING_DIM
        )
        self._maia2_feature_embedding_2 = torch.nn.Linear(
            MAIA2_FEATURE_INTERMEDIATE_EMBEDDING_DIM, MAIA2_FEATURE_EMBEDDING_DIM
        )

        self.rnn = nn.RNN(
            input_size=DIM,
            hidden_size=DIM,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )

        self.fc1 = torch.nn.Linear(
            DIM + MOVE_EMBED_DIM + MAIA2_FEATURE_EMBEDDING_DIM, DIM
        )
        self.fc2 = torch.nn.Linear(DIM, 512)
        self.fc3 = torch.nn.Linear(512, 1)
        self.dropoutrnn = torch.nn.Dropout(0.2)
        self.dropout1 = torch.nn.Dropout(0.15)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, boards, _, num_moves, maia2_features):
        BSZ, LEN = boards.shape[0], boards.shape[1]
        boards = boards.reshape(-1, 112, 8, 8)
        # maia2_boards = maia2_boards.reshape(-1, 18, 8, 8)
        h_list = []
        for _board_embedder, _dim_reducer in zip(
            self._board_embedders, self._board_embedding_dim_reducers
        ):
            board_batch = _board_embedder(boards)
            board_batch = board_batch.reshape(BSZ, LEN, -1)
            board_batch = _dim_reducer(board_batch)
            h_list.append(board_batch)

        h = torch.stack(h_list).mean(axis=0)

        packed = nn.utils.rnn.pack_padded_sequence(
            h,
            torch.clamp(num_moves, min=0, max=10).cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden = self.rnn(packed)

        moves_embedding = self._num_move_embed(num_moves)
        maia2_feature_embedding_intermediate = self._maia2_feature_embedding_1(
            maia2_features
        )
        maia2_feature_embedding_intermediate = torch.nn.functional.relu(
            maia2_feature_embedding_intermediate
        )
        maia2_feature_embedding = self._maia2_feature_embedding_2(
            maia2_feature_embedding_intermediate
        )
        maia2_feature_embedding = torch.nn.functional.relu(maia2_feature_embedding)

        x = self.fc1(
            torch.cat([hidden[-1], moves_embedding, maia2_feature_embedding], dim=-1)
        )
        x = torch.nn.functional.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = torch.nn.functional.relu(x)
        x = self.fc3(x)
        return x

    def serialize(self, path):
        torch.save(
            {
                "model": self.state_dict(),
                "leela": [
                    embedder.serialize_to_dict() for embedder in self._board_embedders
                ],
            },
            path,
        )

    @classmethod
    def deserialize(cls, path):
        data = torch.load(path, map_location="cpu", weights_only=False)
        leela_types = []
        for m in data["leela"]:
            if "self_elo" in m:
                leela_types.append((m["model_type"], m["self_elo"], m["oppo_elo"]))
            else:
                leela_types.append(m["model_type"])
        model = cls(leela_model_types=leela_types)
        model.load_state_dict(data["model"])
        return model


def evaluate(model, eval_data, device=torch.device(0)):
    valid_loss = torch.nn.MSELoss(reduction="sum")
    model.eval()

    tot_loss = 0.0
    for (
        boards,
        _, # maia2_boards, no longer used
        num_moves,
        maia2_features,
        ratings,
    ) in eval_data:
        boards = boards.to(device)
        num_moves = num_moves.to(device)
        maia2_features = maia2_features.to(device)
        ratings = ratings.to(device)
        ratings = ratings * CLEAN_STD + CLEAN_MEAN
        pred = model(boards, _, num_moves, maia2_features)
        pred = pred * CLEAN_STD + CLEAN_MEAN

        loss = valid_loss(
            torch.nan_to_num(pred.reshape(-1), nan=1510.0), ratings.reshape(-1)
        )
        tot_loss += loss.item()
        del (
            boards,
            # maia2_boards,
            num_moves,
            maia2_features,
            ratings,
        )
    model.train()
    return tot_loss / len(eval_data.dataset)


class WrappedMSELoss(torch.nn.Module):
    def __init__(self):
        super(WrappedMSELoss, self).__init__()
        self._mse = torch.nn.MSELoss()

    def forward(self, pred, target):
        residues = torch.abs(
            (pred * CLEAN_STD + CLEAN_MEAN) - (target * CLEAN_STD + CLEAN_MEAN)
        )
        mse_indices = (residues > 200.0) & (residues <= 450.0)
        mae_indices = residues <= 200.0
        quartic_indices = residues > 450.0
        return (
            self._mse(pred, target),
            self._mse(
                pred.detach() * CLEAN_STD + CLEAN_MEAN,
                target.detach() * CLEAN_STD + CLEAN_MEAN,
            ),
            mae_indices.sum(),
            mse_indices.sum(),
            quartic_indices.sum(),
        )


def simple_train_loop(
    model,
    train_dataset,
    valid_dataset,
    epochs=5,
    device=torch.device(0),
    optimizer=None,
    experiment_name="experiment",
):
    tot_loss = 0.0
    BEST_LOSS = 100000000000000000000.0
    mse_loss_fn = WrappedMSELoss()
    train_dataloader = DataLoader(
        train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=4
    )
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=256, shuffle=True, num_workers=4
    )

    model.train()
    logger.info(f"PARAMETERS: {model.count_parameters()}")
    if not optimizer:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = InverseSquareRootSchedule(
        optimizer, warmup_init_lr=5e-8, warmup_updates=4000, lr=3e-4
    )
    for epoch in range(epochs):

        logger.info(f"EPOCH {epoch}")
        if epoch != 0:
            logger.info("EVALUATE")
            valid_loss = evaluate(model, valid_dataloader, device=device)
            logger.info(f"VALID LOSS: {valid_loss}")
            if valid_loss < BEST_LOSS:
                logger.info(f"NEW BEST LOSS. SERIALIZING.")
                BEST_LOSS = valid_loss
                model.serialize(
                    f"experiment_models/{experiment_name}/{experiment_name}.best.ckpt"
                )
            model.train()
        tot_metric_loss, tot_mse_loss = 0.0, 0.0
        tot_mae_count, tot_mse_count, tot_quartic_count = 0, 0, 0
        for idx, (
            boards,
            _, # maia2_boards,
            num_moves,
            maia2_features,
            ratings,
        ) in enumerate(train_dataloader):
            optimizer.zero_grad()
            boards = boards.to(device)
            # maia2_boards = maia2_boards.to(device)
            num_moves = num_moves.to(device)
            maia2_features = maia2_features.to(device)
            ratings = ratings.to(device)
            pred = model(boards, None, num_moves, maia2_features)
            metric_loss, mse_loss, mae_count, mse_count, quartic_count = mse_loss_fn(
                torch.nan_to_num(pred.reshape(-1), nan=1510.0), ratings.reshape(-1)
            )

            metric_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            tot_metric_loss += metric_loss.item()
            tot_mse_loss += mse_loss.item()
            tot_mae_count += mae_count.item()
            tot_mse_count += mse_count.item()
            tot_quartic_count += quartic_count.item()
            if idx > 0 and idx % 100 == 0:
                if idx % 1000 == 0:
                    logger.info(pred.detach().reshape(-1) * CLEAN_STD + CLEAN_MEAN)
                    logger.info(ratings.reshape(-1) * CLEAN_STD + CLEAN_MEAN)
                    logger.info(
                        f"metric_loss: {metric_loss:0.3f}, mse_loss: {mse_loss.item():0.3f}, counts: {mae_count.item(), mse_count.item(), quartic_count.item()}"
                    )
                if idx % 2000 == 0:
                    valid_loss = evaluate(model, valid_dataloader, device=device)
                    logger.info(f"VALID LOSS: {valid_loss}")
                    if valid_loss < BEST_LOSS:
                        logger.info(f"NEW BEST LOSS. SERIALIZING.")
                        BEST_LOSS = valid_loss
                        model.serialize(
                            f"experiment_models/{experiment_name}/{experiment_name}.best.ckpt"
                        )
                    model.train()
                s = tot_mae_count + tot_mse_count + tot_quartic_count
                a, b, c = tot_mae_count / s, tot_mse_count / s, tot_quartic_count / s
                logger.info(
                    f"batch {idx} / {len(train_dataloader)}, tot_metric_loss: {tot_metric_loss / (idx + 1):0.3f}, tot_mse_loss: {tot_mse_loss / (idx+1):0.3f}, counts: {a:0.4f}, {b:0.4f}, {c:0.4f} lr: {scheduler.get_last_lr()[0]:0.10f}"
                )
            del (
                boards,
                # maia2_boards,
                num_moves,
                maia2_features,
                ratings,
            )
    return model, optimizer


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    torch.autograd.set_detect_anomaly(True)

    parser.add_argument(
        "--experiment-name",
        type=str,
        help="the name of the experiment for the log file",
        default="experiment",
    )
    parser.add_argument(
        "--epochs", type=int, help="number of training epochs", default=20
    )
    parser.add_argument("--device", type=int, help="gpu device number", default=0)
    parser.add_argument(
        "--model-types",
        type=str,
        help="model types must be [small, med, large, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]",
        default="small",
    )
    parser.add_argument(
        "--flip-boards", type=float, default=0.0, help="randomly flips puzzle sides"
    )
    parser.add_argument(
        "--use-cleaned-datasets",
        action="store_true",
        help="use the up-to-date, cleaned datasets",
    )
    parser.add_argument(
        "--use-maia2-features",
        action="store_true",
        help="use the maia2 features, like the win probs",
    )
    args = parser.parse_args()

    args.model_types = args.model_types.strip().split(",")

    model_types = []
    for model_type in args.model_types:
        if model_type == "small":
            model_types.append(leela_utils.LEELA_TYPE.SMALL)
        elif model_type == "med":
            model_types.append(leela_utils.LEELA_TYPE.MED)
        elif model_type == "large":
            model_types.append(leela_utils.LEELA_TYPE.LARGE)
        elif model_type == "1100":
            model_types.append(maia_utils.MAIA_TYPE.MAIA_1100)
        elif model_type == "1200":
            model_types.append(maia_utils.MAIA_TYPE.MAIA_1200)
        elif model_type == "1300":
            model_types.append(maia_utils.MAIA_TYPE.MAIA_1300)
        elif model_type == "1400":
            model_types.append(maia_utils.MAIA_TYPE.MAIA_1400)
        elif model_type == "1500":
            model_types.append(maia_utils.MAIA_TYPE.MAIA_1500)
        elif model_type == "1600":
            model_types.append(maia_utils.MAIA_TYPE.MAIA_1600)
        elif model_type == "1700":
            model_types.append(maia_utils.MAIA_TYPE.MAIA_1700)
        elif model_type == "1800":
            model_types.append(maia_utils.MAIA_TYPE.MAIA_1800)
        elif model_type == "1900":
            model_types.append(maia_utils.MAIA_TYPE.MAIA_1900)
        else:
            raise ValueError(
                "--model-type, must be [small, med, large, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, m2{rapid|blitz}_{1100-1900}_{1100-1900}]"
            )

    Path(f"experiment_models/{args.experiment_name}").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=f"experiment_models/{args.experiment_name}/{args.experiment_name}.log",
        level=logging.INFO,
        format="%(asctime)s: %(message)s",  # Format the log messages
        datefmt="%m-%d-%y %H:%M:%S",  # Format for the timestamp
    )

    DEVICE = torch.device(args.device)
    baseline = BatchedBaseline(leela_model_types=model_types).to(DEVICE)
    train_dataset = data_loader.load_data(
        "train",
        flip_boards=args.flip_boards,
        use_cleaned_dataset=args.use_cleaned_datasets,
        use_maia2_features=args.use_maia2_features,
    )
    valid_dataset = data_loader.load_data(
        "valid",
        use_cleaned_dataset=args.use_cleaned_datasets,
        use_maia2_features=args.use_maia2_features,
    )
    test_dataset = data_loader.load_data(
        "test",
        use_cleaned_dataset=args.use_cleaned_datasets,
        use_maia2_features=args.use_maia2_features,
    )
    baseline, optimizer = simple_train_loop(
        baseline,
        train_dataset,
        valid_dataset,
        device=DEVICE,
        epochs=args.epochs,
        experiment_name=args.experiment_name,
    )
