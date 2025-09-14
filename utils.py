from leela_board import LeelaBoard
import chess
import torch
import math


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


from common import CLEAN_STD, CLEAN_MEAN
def _rescale_score(x, scale=0.9, STD=CLEAN_STD, MEAN=CLEAN_MEAN):
    if x > 1900 or x < 1100:
        x = (x - MEAN) / STD
        x *= scale

        return x * STD + MEAN
    return x

def merge_old_and_new_ieee(old, new):
    old_dataset, old_sol = open("datasets/old_ieee_dataset.csv", 'r'), open(old, 'r')
    old_mapping = {}
    old_dataset.readline()
    for (example, score) in zip(old_dataset, old_sol):
        example = example.strip().split(',')[1]
        score = int(score.strip())
        if example not in old_mapping:
            old_mapping[example] = score

    new_dataset, new_sol = open("datasets/testing_data_cropped.csv", 'r'), open(new, 'r')
    new_mapping = {}

    new_dataset.readline()
    for (example, score) in zip(new_dataset, new_sol):
        example = example.strip().split(',')[1]
        score = int(score.strip())
        if example in old_mapping:
            print(int((old_mapping[example] + score) / 2))
        else:
            print(score)

    # for example in new_mapping:
    #     if example in old_mapping:
    #         print(int((old_mapping[example] + new_mapping[example]) / 2))

MEAN = 1510
def rescale(x, hi=400, lo=400, D = 800):
  if x > MEAN:
    return max(MEAN, x - hi*min(1, ((x - MEAN)/D)**4))
  else:
    return min(MEAN, x + lo*min(1, ((x - MEAN)/D)**4))

# def map_points_power(self, points, power=0.5):
#     """
#     Map points using power function for non-linear shrinking.
    
#     Parameters:
#     points (array-like): Points from the original distribution
#     power (float): Power parameter (0 < power < 1 for shrinking)
    
#     Returns:
#     numpy.ndarray: Transformed points
#     """
#     points = np.array(points)
    
#     # Calculate distance from mean
#     distance = points - self.original_mean
    
#     # Apply power transformation while preserving sign
#     sign = np.sign(distance)
#     shrunk_distance = sign * np.abs(distance)**power
    
#     # Adjust for target distribution
#     scale_factor = self.target_std / (self.original_std**power)
#     transformed = shrunk_distance * scale_factor + self.target_mean
    
#     return transformed

def rescale_power(x, pow=0.5):
    if x > 1510:
        d = x - 1510
        shrunk = d ** pow
        scale = 250 / (434**pow)
        return int(shrunk * scale + 1510)
    else:
        d = x - 1510
        sign = -1 if d < 0 else 1
        shrunk = sign * math.fabs(d) ** pow
        scale = 250 / (434**pow)
        return int(shrunk * scale + 1510)

def rescale_to_mean_pow(value, mean, std, power=1.3):
    """
    Rescale a value to move it towards the mean, with farther values 
    moving more than closer values.
    
    Parameters:
    - value: the value to rescale
    - mean: mean of the original dataset
    - std: standard deviation of the original dataset  
    - power: compression factor (0 < power < 1). Lower = more compression
    
    Returns:
    - rescaled value
    """
    if std == 0:
        return mean
    
    # Normalize distance from mean
    normalized_distance = (value - mean) / std
    
    # Compress the distance while preserving sign
    if normalized_distance >= 0:
        compressed_distance = normalized_distance ** power
    else:
        compressed_distance = -(abs(normalized_distance) ** power)
    
    # Scale back to original units
    return mean + compressed_distance * std

def shrink_toward_mean_k(x, mean, std, k=1.0):
    """
    Pull x toward 'mean' with strength controlled by 'k'.
    z = (x - mean)/std
    z_shrunk = (k * z) / (k + |z|)
    -> when k=1.0, this is the same as z/(1 + |z|).
    Larger k means less shrinkage; smaller k means more aggressive shrinkage.
    """
    if std == 0:
        return mean

    z = (x - mean) / std
    z_shrunk = (k * z) / (k + abs(z))
    return mean + std * z_shrunk

def rescale_paper(x):
    if x < 600:
        return x + 400
    elif x < 700: 
        return x + 300
    elif x < 800:
        return x + 200
    elif x < 900:
        return x + 100
    elif x < 1000:
        return x + 50
    if x > 2600:
        return x - 300
    elif x > 2500:
        return x - 200
    elif x > 2400:
        return x - 100
    elif x > 2300:
        return x - 50
    return x

def tanh_h_scale(x, mean, std, alpha=1.0):
    """
    alpha < 1.0 → gentler squeeze (outliers not clamped as hard)
    alpha > 1.0 → stronger squeeze
    """
    if std == 0:
        return mean
    z = (x - mean) / std
    return int(mean + std * math.tanh(alpha * z))

def sqrt_compress(value, mean, scale=0.5):
    deviation = value - mean
    if deviation >= 0:
        compressed_deviation = (deviation ** 0.5) * scale
    else:
        compressed_deviation = -((abs(deviation)) ** 0.5) * scale
    return int(mean + compressed_deviation)



def rescale_log(x, MEAN=1400, STD=350, OFFSET=0):
    # return int(x)
    s = (x - MEAN) / STD
    s += 0.000000001

    sign = x > MEAN
    if sign:
        scale = 0.95
    else:
        
        scale = 0.5
    
    out = int(STD*scale*s + MEAN + OFFSET - (100 if sign else 0))
    # if out > 2000: out -= 50
    return out
if __name__ == "__main__":
    import sys

    # files = sys.argv[1:-3]
    # lo, hi, D = int(sys.argv[-3]), int(sys.argv[-2]), int(sys.argv[-1])
    files = sys.argv[1:]

    # scale = float(sys.argv[-1])
    vals = []
    with open(files[0], "r") as file:
        for line in file:
            vals.append(int(line.strip()))
    for f in files[1:]:
        with open(f, "r") as file:
            for (idx, line) in enumerate(file):
                vals[idx] += int(line.strip())
    out = []
    for v in vals:
        v = int(v / len(files))
        # print(v)
        print(int(rescale_log(v)))
        
        # a = abs(v - 1510)
        # sign = v > 1510
        # v = (v - 1510) / 424
        # scale = 1.0

        # if sign:
        #     if a < 200:
        #         scale = 0.38
        #     elif a < 300:
        #         scale = 0.4
        #     elif a < 500:
        #         scale = 0.65
        #     elif a < 800:
        #         scale = 0.85
        # else:
        #     scale = 0.7
        # v = v * scale * 424 + 1510

        # out.append(int(v) + 25)
        # out.append(tanh_h_scale(v, 1510, 424, alpha=0.5))

        # new_v = (v - 1510) / 424
        # new_v = new_v * multiplier * 424 + 1510
        # print(int(v) + 25)
        # out.append(int(new_v))
    # print(' '.join(map(str, out)))
    # print(out)
    # for w in out:
    #     print(w)

            # print(int(rescale(v / len(files), 400, 400, 800)))
        # else:
        #     # print('nah')
        #     print(int(v / len(files)))
    # for idx, val in enumerate(vals):
    #     vals[idx] = _rescale_score(val / len(files), scale=scale)
    #     print(f"{int(vals[idx])}")

    # old, new = sys.argv[1], sys.argv[2]
    # merge_old_and_new_ieee(old, new)