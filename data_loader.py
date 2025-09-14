from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import utils
import torch
import random
import maia2.utils

import chess

from common import CLEAN_MEAN, CLEAN_STD

MOVE_BUCKETS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
MAX_MOVES = 10

def get_move_bucket(moves):
    if moves > 15:
        return len(MOVE_BUCKETS) - 1
    return moves


class SimpleBaselineDataset(Dataset):
    def __init__(
        self,
        puzzles_file,
        filter=False,
        regularize=False,
        flip_boards=0.0,
        has_maia2_probs=False,
    ):
        self.puzzles = pd.read_csv(puzzles_file)
        if filter:
            self.puzzles = self.puzzles.loc[
                (self.puzzles["RatingDeviation"] <= 100.0)
                & (self.puzzles["NbPlays"] >= 500)
                | (self.puzzles["Rating"] <= 1000)
                | (self.puzzles["Rating"] >= 2000)
            ]
        self._regularize = regularize
        self.flip_boards = flip_boards

    def __len__(self):
        return len(self.puzzles)

    def __getitem__(self, idx):
        example = self.puzzles.iloc[idx]
        fen = example[1]
        if self._regularize:
            rating = example[3] + (random.random() * 2 - 1) * 20
        else:
            rating = example[3]
        moves = example[2]
        moves = moves.split()

        if random.random() < self.flip_boards:
            fen, moves = utils.flip_board(fen, moves)

        board = utils.ChessBoard(fen)
        board.move(moves[0])

        board_size = board.t.shape
        boards = torch.zeros(MAX_MOVES, *board_size)
        boards[0] = board.t

        maia2_board = chess.Board(fen)
        maia2_board.push_san(moves[0])

        maia2_boards = torch.zeros(MAX_MOVES, 18, 8, 8)
        maia2_boards[0] = maia2.utils.board_to_tensor(maia2_board)

        for i in range(1, min(len(moves), MAX_MOVES)):
            board.move(moves[i])
            boards[i] = board.t

            maia2_board.push_san(moves[i])
            maia2_boards[i] = maia2.utils.board_to_tensor(maia2_board)

        moves_bucket = get_move_bucket(len(moves))

        if "success_prob_blitz_1550" in example:
            maia_features = torch.from_numpy(example[10:].values.astype(np.float32))
        else:
            maia_features = False
        return (
            boards,
            maia2_boards,
            torch.tensor(moves_bucket, dtype=torch.int64),
            maia_features,
            torch.tensor((rating - CLEAN_MEAN) / CLEAN_STD, dtype=torch.float32),
        )


def load_data(
    dataset_name,
    regularize_train_dataset=False,
    flip_boards=0.0,
    use_cleaned_dataset=False,
    use_maia2_features=False,
):
    if use_maia2_features:
        if dataset_name == "train":
            return SimpleBaselineDataset(
                (
                    "datasets/cleaned_maia2_features_train_set.csv"
                    if use_cleaned_dataset
                    else "datasets/maia2_features_train_set.csv"
                ),
                regularize=regularize_train_dataset,
                flip_boards=flip_boards,
            )
        elif dataset_name == "valid":
            return SimpleBaselineDataset(
                (
                    "datasets/cleaned_maia2_features_validation_set.csv"
                    if use_cleaned_dataset
                    else "datasets/maia2_features_validation_set.csv"
                ),
                flip_boards=flip_boards,
            )
        elif dataset_name == "test":
            return SimpleBaselineDataset(
                (
                    "datasets/cleaned_maia2_features_test_set.csv"
                    if use_cleaned_dataset
                    else "datasets/maia2_features_test_set.csv"
                ),
                flip_boards=flip_boards,
            )
    else:

        if dataset_name == "train":
            return SimpleBaselineDataset(
                (
                    "datasets/cleaned_train_set.csv"
                    if use_cleaned_dataset
                    else "datasets/train_set.csv"
                ),
                regularize=regularize_train_dataset,
                flip_boards=flip_boards,
            )
        elif dataset_name == "valid":
            return SimpleBaselineDataset(
                (
                    "datasets/cleaned_validation_set.csv"
                    if use_cleaned_dataset
                    else "datasets/validation_set.csv"
                ),
                flip_boards=flip_boards,
            )
        elif dataset_name == "test":
            return SimpleBaselineDataset(
                (
                    "datasets/cleaned_test_set.csv"
                    if use_cleaned_dataset
                    else "datasets/test_set.csv"
                ),
                flip_boards=flip_boards,
            )
        elif dataset_name == "ieee":
            return IEEEDataset(
                "datasets/ieee_dataset.csv",
                # flip_boards=flip_boards
            )
        else:
            raise ValueError(
                f"invalid dataset name {dataset_name}, must be [train, valid, test, ieee]"
            )


def load_ieee(flip_boards=0.0):
    return IEEEDataset("datasets/testing_data_cropped.csv")
    # return IEEEDataset("datasets/ieee_dataset.csv", flip_boards=flip_boards)


class IEEEDataset(Dataset):
    def __init__(self, puzzles_file, filter=False, regularize=False):
        self.puzzles = pd.read_csv(puzzles_file)

    def __len__(self):
        return len(self.puzzles)

    def __getitem__(self, idx):
        example = self.puzzles.iloc[idx]
        fen = example[1]
        moves = example[2]
        moves = moves.split()

        board = utils.ChessBoard(fen)
        board.move(moves[0])

        board_size = board.t.shape
        boards = torch.zeros(MAX_MOVES, *board_size)
        boards[0] = board.t

        maia2_board = chess.Board(fen)
        maia2_board.push_san(moves[0])

        maia2_boards = torch.zeros(MAX_MOVES, 18, 8, 8)
        maia2_boards[0] = maia2.utils.board_to_tensor(maia2_board)

        for i in range(1, min(len(moves), MAX_MOVES)):
            board.move(moves[i])
            boards[i] = board.t

            maia2_board.push_san(moves[i])
            maia2_boards[i] = maia2.utils.board_to_tensor(maia2_board)

        moves_bucket = get_move_bucket(len(moves))

        if "success_prob_blitz_1550" in example:
            maia_features = torch.from_numpy(example[3:].values.astype(np.float32))
        else:
            maia_features = False

        return (
            boards,
            maia2_boards,
            torch.tensor(moves_bucket, dtype=torch.int64),
            maia_features,
            False,
        )
