import torch
from torch.utils.data import DataLoader
from scheduler import InverseSquareRootSchedule

import data_loader
from data_loader import MOVE_BUCKETS
import leela_utils
import maia_utils

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)
DIM, MOVE_EMBED_DIM = 1024, 128


class EnsembleBaseline(torch.nn.Module):
    def __init__(self, leela_model_types=[leela_utils.LEELA_TYPE.SMALL]):
        super(EnsembleBaseline, self).__init__()
        self._leela_model_types = leela_model_types

        self._board_embedders = []
        self._board_embedding_dims = []
        for leela_model_type in self._leela_model_types:
            # This should handle both Leela and Maia models. Not sure if this is the "most optimal" way to do this.
            if isinstance(leela_model_type, leela_utils.LEELA_TYPE):
                self._board_embedders.append(
                    leela_utils.LeelaEmbedder(leela_model_type)
                )
            else:
                self._board_embedders.append(maia_utils.MaiaEmbedder(leela_model_type))
            self._board_embedding_dims.append(self._board_embedders[-1].embed_size())
        self._board_embedders = torch.nn.ParameterList(self._board_embedders)

        self._num_move_embed = torch.nn.Embedding(len(MOVE_BUCKETS), MOVE_EMBED_DIM)
        self._start = torch.nn.Parameter(data=torch.Tensor(DIM), requires_grad=True)
        torch.nn.init.uniform_(self._start.data)

        self.combiner_a, self.combiner_b = torch.nn.Linear(DIM, DIM), torch.nn.Linear(
            DIM, DIM
        )
        self.board_embedding_dim_reducers = {}
        for size in set(self._board_embedding_dims):
            self.board_embedding_dim_reducers[str(size)] = torch.nn.Linear(size, DIM)
        self.board_embedding_dim_reducers = torch.nn.ParameterDict(
            self.board_embedding_dim_reducers
        )
        self.fc1 = torch.nn.Linear(DIM + MOVE_EMBED_DIM, DIM)
        self.fc2 = torch.nn.Linear(DIM, 512)
        self.fc3 = torch.nn.Linear(512, 1)
        self.dropoutrnn = torch.nn.Dropout(0.2)
        self.dropout1 = torch.nn.Dropout(0.15)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, boards, flags, num_moves):
        third_flags = flags[0]
        fourth_flags = flags[1]
        fifth_flag = flags[2]
        sixth_flag = flags[3]
        seventh_flag = flags[4]
        eighth_flag = flags[5]
        ninth_flag = flags[6]
        tenth_flag = flags[7]

        h_list = []
        for _board_embedder, size in zip(
            self._board_embedders, self._board_embedding_dims
        ):
            first_board_embedding = _board_embedder(boards[0])
            second_board_embedding = _board_embedder(boards[1])
            third_board_embedding = _board_embedder(boards[2])
            fourth_board_embedding = _board_embedder(boards[3])
            fifth_board_embedding = _board_embedder(boards[4])
            sixth_board_embedding = _board_embedder(boards[5])
            seventh_board_embedding = _board_embedder(boards[6])
            eighth_board_embedding = _board_embedder(boards[7])
            ninth_board_embedding = _board_embedder(boards[8])
            tenth_board_embedding = _board_embedder(boards[9])

            third_board_embedding[torch.logical_not(third_flags)] = torch.zeros_like(
                first_board_embedding[0, :]
            )
            fourth_board_embedding[torch.logical_not(fourth_flags)] = torch.zeros_like(
                first_board_embedding[0, :]
            )
            fifth_board_embedding[torch.logical_not(fifth_flag)] = torch.zeros_like(
                first_board_embedding[0, :]
            )
            sixth_board_embedding[torch.logical_not(sixth_flag)] = torch.zeros_like(
                first_board_embedding[0, :]
            )
            seventh_board_embedding[torch.logical_not(seventh_flag)] = torch.zeros_like(
                first_board_embedding[0, :]
            )
            eighth_board_embedding[torch.logical_not(eighth_flag)] = torch.zeros_like(
                first_board_embedding[0, :]
            )
            ninth_board_embedding[torch.logical_not(ninth_flag)] = torch.zeros_like(
                first_board_embedding[0, :]
            )
            tenth_board_embedding[torch.logical_not(tenth_flag)] = torch.zeros_like(
                first_board_embedding[0, :]
            )

            board_embedding_dim_reducer = self.board_embedding_dim_reducers[str(size)]

            first_board_embedding = board_embedding_dim_reducer(first_board_embedding)
            first_board_embedding = torch.nn.functional.relu(first_board_embedding)

            second_board_embedding = board_embedding_dim_reducer(second_board_embedding)
            second_board_embedding = torch.nn.functional.relu(second_board_embedding)

            third_board_embedding = board_embedding_dim_reducer(third_board_embedding)
            third_board_embedding = torch.nn.functional.relu(third_board_embedding)

            fourth_board_embedding = board_embedding_dim_reducer(fourth_board_embedding)
            fourth_board_embedding = torch.nn.functional.relu(fourth_board_embedding)

            fifth_board_embedding = board_embedding_dim_reducer(fifth_board_embedding)
            fifth_board_embedding = torch.nn.functional.relu(fifth_board_embedding)

            sixth_board_embedding = board_embedding_dim_reducer(sixth_board_embedding)
            sixth_board_embedding = torch.nn.functional.relu(sixth_board_embedding)

            seventh_board_embedding = board_embedding_dim_reducer(
                seventh_board_embedding
            )
            seventh_board_embedding = torch.nn.functional.relu(seventh_board_embedding)

            eighth_board_embedding = board_embedding_dim_reducer(eighth_board_embedding)
            eighth_board_embedding = torch.nn.functional.relu(eighth_board_embedding)

            ninth_board_embedding = board_embedding_dim_reducer(ninth_board_embedding)
            ninth_board_embedding = torch.nn.functional.relu(ninth_board_embedding)

            tenth_board_embedding = board_embedding_dim_reducer(tenth_board_embedding)
            tenth_board_embedding = torch.nn.functional.relu(tenth_board_embedding)

            h = torch.nn.functional.tanh(
                self.combiner_a(self._start) + self.combiner_b(first_board_embedding)
            )
            h = torch.nn.functional.tanh(
                self.combiner_a(h) + self.combiner_b(second_board_embedding)
            )
            h = torch.nn.functional.tanh(
                self.combiner_a(h) + self.combiner_b(third_board_embedding)
            )
            h = torch.nn.functional.tanh(
                self.combiner_a(h) + self.combiner_b(fourth_board_embedding)
            )
            h = torch.nn.functional.tanh(
                self.combiner_a(h) + self.combiner_b(fifth_board_embedding)
            )
            h = torch.nn.functional.tanh(
                self.combiner_a(h) + self.combiner_b(sixth_board_embedding)
            )
            h = torch.nn.functional.tanh(
                self.combiner_a(h) + self.combiner_b(seventh_board_embedding)
            )
            h = torch.nn.functional.tanh(
                self.combiner_a(h) + self.combiner_b(eighth_board_embedding)
            )
            h = torch.nn.functional.tanh(
                self.combiner_a(h) + self.combiner_b(ninth_board_embedding)
            )
            h = torch.nn.functional.tanh(
                self.combiner_a(h) + self.combiner_b(tenth_board_embedding)
            )
            h = self.dropoutrnn(h)

            h_list.append(h)
        h = torch.stack(h_list).mean(axis=0)

        moves_embedding = self._num_move_embed(num_moves)

        x = self.fc1(torch.cat([h, moves_embedding], dim=-1))
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
        data = torch.load(path, map_location="cpu")
        leela_types = [m["model_type"] for m in data["leela"]]
        model = cls(leela_model_types=leela_types)
        model.load_state_dict(data["model"])
        return model


def evaluate(model, eval_data, device=torch.device(0)):
    valid_loss = torch.nn.MSELoss(reduction="sum")
    model.eval()

    tot_loss = 0.0
    for (
        first_board,
        second_board,
        third_board,
        fourth_board,
        fifth_board,
        sixth_board,
        seventh_board,
        eighth_board,
        ninth_board,
        tenth_board,
        third_flag,
        fourth_flag,
        fifth_flag,
        sixth_flag,
        seventh_flag,
        eighth_flag,
        ninth_flag,
        tenth_flag,
        _,  # was themes
        num_moves,
        _,  # was rating_buckets
        ratings,
    ) in eval_data:
        (
            first_board,
            second_board,
            third_board,
            fourth_board,
            fifth_board,
            sixth_board,
            seventh_board,
            eighth_board,
            ninth_board,
            tenth_board,
        ) = (
            first_board.to(device),
            second_board.to(device),
            third_board.to(device),
            fourth_board.to(device),
            fifth_board.to(device),
            sixth_board.to(device),
            seventh_board.to(device),
            eighth_board.to(device),
            ninth_board.to(device),
            tenth_board.to(device),
        )
        (
            third_flag,
            fourth_flag,
            fifth_flag,
            sixth_flag,
            seventh_flag,
            eighth_flag,
            ninth_flag,
            tenth_flag,
        ) = (
            third_flag.to(device),
            fourth_flag.to(device),
            fifth_flag.to(device),
            sixth_flag.to(device),
            seventh_flag.to(device),
            eighth_flag.to(device),
            ninth_flag.to(device),
            tenth_flag.to(device),
        )
        num_moves = num_moves.to(device)
        # themes = themes.to(device)
        # rating_buckets = rating_buckets.to(device)
        ratings = ratings.to(device)
        pred = model(
            (
                first_board,
                second_board,
                third_board,
                fourth_board,
                fifth_board,
                sixth_board,
                seventh_board,
                eighth_board,
                ninth_board,
                tenth_board,
            ),
            (
                third_flag,
                fourth_flag,
                fifth_flag,
                sixth_flag,
                seventh_flag,
                eighth_flag,
                ninth_flag,
                tenth_flag,
            ),
            num_moves,
        )

        loss = valid_loss(pred.reshape(-1), ratings.reshape(-1))
        tot_loss += loss.item()
        del (
            first_board,
            second_board,
            third_board,
            fourth_board,
            fifth_board,
            sixth_board,
            seventh_board,
            eighth_board,
            ninth_board,
            tenth_board,
            third_flag,
            fourth_flag,
            fifth_flag,
            sixth_flag,
            seventh_flag,
            eighth_flag,
            ninth_flag,
            tenth_flag,
            _,  # was themes
            num_moves,
            ratings,
            # rating_buckets,
        )
    model.train()
    return tot_loss / len(eval_data.dataset)


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
    mse_loss_fn = torch.nn.MSELoss()
    train_dataloader = DataLoader(
        train_dataset, batch_size=256, shuffle=True, num_workers=4
    )
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=256, shuffle=True, num_workers=4
    )

    model.train()
    logger.info(f"PARAMETERS: {model.count_parameters()}")
    if not optimizer:
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    scheduler = InverseSquareRootSchedule(
        optimizer, warmup_init_lr=5e-8, warmup_updates=8000, lr=5e-5
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
        tot_loss = 0.0
        for idx, (
            first_board,
            second_board,
            third_board,
            fourth_board,
            fifth_board,
            sixth_board,
            seventh_board,
            eighth_board,
            ninth_board,
            tenth_board,
            third_flag,
            fourth_flag,
            fifth_flag,
            sixth_flag,
            seventh_flag,
            eighth_flag,
            ninth_flag,
            tenth_flag,
            _,  # was themes
            num_moves,
            _,  # was rating buckets
            ratings,
        ) in enumerate(train_dataloader):
            optimizer.zero_grad()
            (
                first_board,
                second_board,
                third_board,
                fourth_board,
                fifth_board,
                sixth_board,
                seventh_board,
                eighth_board,
                ninth_board,
                tenth_board,
            ) = (
                first_board.to(device),
                second_board.to(device),
                third_board.to(device),
                fourth_board.to(device),
                fifth_board.to(device),
                sixth_board.to(device),
                seventh_board.to(device),
                eighth_board.to(device),
                ninth_board.to(device),
                tenth_board.to(device),
            )
            (
                third_flag,
                fourth_flag,
                fifth_flag,
                sixth_flag,
                seventh_flag,
                eighth_flag,
                ninth_flag,
                tenth_flag,
            ) = (
                third_flag.to(device),
                fourth_flag.to(device),
                fifth_flag.to(device),
                sixth_flag.to(device),
                seventh_flag.to(device),
                eighth_flag.to(device),
                ninth_flag.to(device),
                tenth_flag.to(device),
            )
            num_moves = num_moves.to(device)
            ratings = ratings.to(device)
            pred = model(
                (
                    first_board,
                    second_board,
                    third_board,
                    fourth_board,
                    fifth_board,
                    sixth_board,
                    seventh_board,
                    eighth_board,
                    ninth_board,
                    tenth_board,
                ),
                (
                    third_flag,
                    fourth_flag,
                    fifth_flag,
                    sixth_flag,
                    seventh_flag,
                    eighth_flag,
                    ninth_flag,
                    tenth_flag,
                ),
                num_moves,
            )
            mse_loss = mse_loss_fn(pred.reshape(-1), ratings.reshape(-1))

            mse_loss.backward()
            optimizer.step()
            scheduler.step()
            tot_loss += mse_loss.item()
            if idx > 0 and idx % 100 == 0:
                if idx % 1000 == 0:
                    logger.info(pred.reshape(-1))
                    logger.info(ratings.reshape(-1))
                    logger.info(f"mse_loss: {mse_loss.item()}")
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

                logger.info(
                    f"batch {idx} / {len(train_dataloader)}, tot_loss: {tot_loss / (idx+1):0.5f}, lr: {scheduler.get_last_lr()[0]:0.10f}"
                )
            del (
                first_board,
                second_board,
                third_board,
                fourth_board,
                fifth_board,
                sixth_board,
                seventh_board,
                eighth_board,
                ninth_board,
                tenth_board,
                third_flag,
                fourth_flag,
                fifth_flag,
                sixth_flag,
                seventh_flag,
                eighth_flag,
                ninth_flag,
                tenth_flag,
                _,  # was themes
                num_moves,
                # rating_buckets,
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
    args = parser.parse_args()

    args.model_types = args.model_types.strip().split(",")
    print(args.model_types)

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
                "--model-type, must be [small, med, large, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]"
            )
    print(model_types)

    Path(f"experiment_models/{args.experiment_name}").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=f"experiment_models/{args.experiment_name}/{args.experiment_name}.log",
        level=logging.INFO,
        format="%(asctime)s: %(message)s",  # Format the log messages
        datefmt="%m-%d-%y %H:%M:%S",  # Format for the timestamp
    )

    DEVICE = torch.device(args.device)
    baseline = EnsembleBaseline(leela_model_types=model_types).to(DEVICE)
    train_dataset = data_loader.load_data(
        "train",
        flip_boards=args.flip_boards,
        use_cleaned_dataset=args.use_cleaned_datasets,
    )
    valid_dataset = data_loader.load_data(
        "valid", use_cleaned_dataset=args.use_cleaned_datasets
    )
    test_dataset = data_loader.load_data(
        "test", use_cleaned_dataset=args.use_cleaned_datasets
    )
    baseline, optimizer = simple_train_loop(
        baseline,
        train_dataset,
        valid_dataset,
        device=DEVICE,
        epochs=args.epochs,
        experiment_name=args.experiment_name,
    )
