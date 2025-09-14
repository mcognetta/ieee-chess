# The `bread emoji` team's submission to the FedCSIS 2025 Predicting Chess Puzzle Difficulty Challenge

This is the implementation of our baseline solution to the challenge. Our solution won second prize and the writeup can be found ~~here~~ (it is not yet public).

The competition website is here: https://knowledgepit.ai/predicting-chess-puzzle-difficulty-2/

Note that this repo is just the training of our base models. The postprocessing rescaling described in our paper (to fit a model trained on the Lichess dataset to the competition dataset distribution) was implemented as a simple transformation mapping on the outputs of these models, and isn't included in this repo.

Our implementation to last year's competition can be found here: https://github.com/mcognetta/ieee-chess/tree/2024-iteration-archive

# Instructions

## Directory setup
Your directory should look something like the following. You can form the split by getting the raw datasets and running `python3 prepare_data.py` (after the dependency installation, since you need `pandas` and `zstandard` to prepare the data).
```
├── batched_embedding_baseline.py
├── common.py
├── data_loader.py
├── datasets
│   ├── cleaned_maia2_features_test_set.csv
│   ├── cleaned_maia2_features_train_set.csv
│   └── cleaned_maia2_features_validation_set.csv
├── leela_board.py
├── leela_flip_test.py
├── leela_utils.py
├── maia_utils.py
├── models
│   ├── leela-small.onnx
│   ├── maia-1100.onnx
│   ├── maia-1300.onnx
│   ├── maia-1500.onnx
│   ├── maia-1700.onnx
│   └── maia-1900.onnx
├── prepare_data.py
├── raw_datasets
│   ├── lichess_db_puzzle.csv
│   └── lichess_db_puzzle.csv.zst
├── scheduler.py
└── utils.py
```

You should also create anb `experiment_models` directory, to store the models:
  - `mkdir experiment_models`

## Dependency installation 

Install the following (recommended to use a conda environment `conda create --name puzzle_rating_prediction python=3.10` then `conda activate puzzle_rating_prediction`):
- pytorch (for linux its something like `conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia` but it depends on your CUDA version; see: https://pytorch.org/get-started/locally/)
- `pip3 install onnx onnx2torch chess pandas numpy zstandard`
- prepare the data `python3 prepare_data.py`
- `mkdir experiment_models`

## Training

To train:
- `python3 batched_embedding_baseline.py --experiment-name example_test --epochs 2 --model-type 1300,small --use-cleaned-datasets --device 0`
- Use `nohup` and `&` to run it in the background for a long time. e.g. `nohup python3 batched_embedding_baseline.py --experiment-name example_test --model-type 1300,small --use-cleaned-datasets --use-maia2-features --epochs 10 --device 0 &`
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

We have provided an example checkpoint for a trained model (using just `Maia 1100`; due to the model sizes, we couldn't upload a larger model; we are investigating another way to do this) for you to use. It is located at `experimental_models/example_1100_model.ckpt`.

You can evaluate it with:

```
python3 evaluate.py --model-path experiment_models/example_1100_model.ckpt --use-cleaned-datasets --
use-maia2-features --ensemble-model
44048.53617384024
```
