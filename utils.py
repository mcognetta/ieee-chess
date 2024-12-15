from leela_board import LeelaBoard
import chess
import torch


def flip_move(move):
    from_square = chess.square_mirror(chess.parse_square(move[:2]))
    to_square = chess.square_mirror(chess.parse_square(move[2:4]))
    promotion = move[4:] if len(move) > 4 else ""
    return chess.square_name(from_square) + chess.square_name(to_square) + promotion


def flip_board(fen, moves):
    temp_board = chess.Board(fen=fen)
    return temp_board.mirror().fen(), [flip_move(move) for move in moves]


# Helper functions
class ChessBoard:
    def __init__(self, fen):  # Create new board from fen
        self.board = LeelaBoard(fen=fen)
        self.t = self.__t()

    def move(self, move):  # Move piece on board ("e2e3")
        self.board.push_uci(move)
        self.t = self.__t()

    def __t(self):  # Set board tensor (private method)
        return torch.from_numpy(self.board.lcz_features()).float()

    def __str__(self):  # Prints board state
        return str(self.board)
