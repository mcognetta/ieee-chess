# The `bread emoji` team's submission to the IEEE BigData Cup 2024 Predicting Chess Puzzle Difficulty Challenge

This is the implementation of our baseline solution to the challenge. Our solution won first prize, and the write up is included as [a PDF in this repo](https://github.com/mcognetta/ieee-chess/blob/master/ieee_bread_emoji_chess_paper.pdf).

The competition website is here: https://knowledgepit.ai/predicting-chess-puzzle-difficulty/

Note that this repo is just the training of our base models. The postprocessing rescaling described in our paper (to fit a model trained on the Lichess dataset to the competition dataset distribution) was implemented as a simple transformation mapping on the outputs of these models, and isn't included in this repo.

# Instructions

## Directory setup
Your directory should look something like the following. You can form the split by getting the raw datasets and running `python3 prepare_data.py` (after the dependency installation, since you need `pandas` and `zstandard` to prepare the data).
```
├── data_loader.py
├── datasets
│   ├── test_set.csv       # possibly a cleaned one also
│   ├── train_set.csv      # possibly a cleaned one also
│   └── validation_set.csv # possibly a cleaned one also
├── leela_board.py
├── leela_flip_test.py
├── leela_utils.py
├── maia_utils.py
├── models
│   ├── leela-large.onnx
│   ├── leela-medium.onnx # omitted due to space
│   └── leela-small.onnx  # omitted due to space
├── prepare_data.py
├── raw_datasets
│   ├── lichess_db_puzzle.csv
│   └── lichess_db_puzzle.csv.zst
├── regression_baseline.py
└── utils.py
```

## Dependency installation 

Install the following (recommended to use a conda environment `conda create --name lichess_puzzle_elo python=3.10` then `conda activate lichess_puzzle_elo`):
- pytorch (for linux its something like `conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia` but it depends on your CUDA version; see: https://pytorch.org/get-started/locally/)
- `pip3 install onnx onnx2torch chess pandas numpy zstandard`
- prepare the data `python3 prepare_data.py`

## Training

To train:
- `python3 regression_baseline.py --experiment-name example_test --epochs 2 --model-type 1300 --use-cleaned-datasets --device 0`
- Use `nohup` and `&` to run it in the background for a long time. e.g. `nohup python3 regression_baseline.py --experiment-name example_test --epochs 10 --device 0 &`
- You can set the underlying leela/maia model via `--model-type {1300,1500,1700,small,med,large}` (but `large` is currently not supported)
  - You can specify more than one with a comma separated list (e.g., `model-type 1300,small`)
  - Other maia models can be used, but we don't package them here

A new directory will appear during the training run:
```
├── experiment_models
│   └── example_test
│       ├── example_test.log
│       └── example_test.best.ckpt
```

- `example_test.log` is a log of the training run, including loss information, etc.
- `example_test.best.ckpt` is the model with the best validation score and is updated throughout training

## An Eval Script

We also have a prepackaged evaluation script to help evaluate corpora:
```
usage: evaluate.py [-h] --model-path MODEL_PATH [--evaluation-corpus EVALUATION_CORPUS] [--device DEVICE] [--verbose]

options:
  -h, --help            show this help message and exit
  --model-path MODEL_PATH
                        the path to the model checkpoint
  --evaluation-corpus EVALUATION_CORPUS
                        the validation corpus. one of {valid, test} (default = test)
  --device DEVICE       gpu device number (default = 0)
  --verbose             print the invidivual predictions and ratings for each puzzle or just the final mse
  --ieee-format         print the rounded int predictions only
  --use-cleaned-datasets
                        use the up-to-date, cleaned datasets
  --ensemble-model      is the model ensembled or not (i.e., if the model contains multiple leela or maia models as board embedders; if it is just one, then
                        this flag should not be set)
  ```

Below is an example invocation:
```
python3 evaluate.py --model-path experiment_models/example_test/example_test.best.ckpt --verbose --device 0
```
