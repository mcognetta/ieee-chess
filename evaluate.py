# >>> _, _, test_dataset = data_loader.load_data()
# >>> model = SimpleBaseline.deserialize("experiment_models/example_test/example_test.best.ckpt")
# >>> model = model.to(DEVICE)
# >>> test_dataloader = DataLoader(test_dataset, batch_size=256)
# >>> evaluate(model, test_dataloader, device=DEVICE)
import torch
from batched_embedding_baseline import *
from common import CLEAN_MEAN, CLEAN_STD

def evaluate(
    model,
    eval_data,
    device=torch.device(0),
    verbose=False,
    ieee_format=False,
    ensemble_model=False,
    rescale_dataloader=False,
):
    model.eval()

    preds_list, ratings_list = [], []
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
        if ensemble_model:
            pred = model(boards, _, num_moves, maia2_features)
        else:
            pred = model(
                boards,
                _,
                num_moves,
            )
        if rescale_dataloader:
            pred = pred * CLEAN_STD + CLEAN_MEAN
            ratings = ratings * CLEAN_STD + CLEAN_MEAN
        preds_list.extend(pred.reshape(-1).tolist())
        ratings_list.extend(ratings.reshape(-1).tolist())
        del (
            boards,
            # maia2_boards,
            num_moves,
            maia2_features,
            ratings,
        )

    if verbose or ieee_format:
        for p, r in zip(preds_list, ratings_list):
            if ieee_format:
                print(f"{int(p)}")
            else:
                print(f"{int(p)} {r:0.4f}")
    else:
        print(
            sum((p - r) ** 2 for (p, r) in zip(preds_list, ratings_list))
            / len(preds_list)
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-path",
        type=str,
        help="the path to the model checkpoint",
        required=True,
    )
    parser.add_argument(
        "--evaluation-corpus",
        type=str,
        help="the validation corpus. one of {valid, test} (default = test)",
        default="test",
    )
    parser.add_argument(
        "--device", type=int, help="gpu device number (default = 0)", default=0
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="print the invidivual predictions and ratings for each puzzle or just the final mse",
    )
    parser.add_argument(
        "--ieee-format",
        action="store_true",
        help="print the rounded int predictions only",
    )
    parser.add_argument(
        "--use-cleaned-datasets",
        action="store_true",
        help="use the up-to-date, cleaned datasets",
    )
    parser.add_argument(
        "--ensemble-model",
        action="store_true",
        help="is the model ensembled or not (i.e., if the model contains multiple leela or maia models as board embedders; if it is just one, then this flag should not be set)",
    )
    parser.add_argument(
        "--use-maia2-features",
        action="store_true",
        help="should maia2 features be included in the dataloader?",
    )
    parser.add_argument(
        "--flip-boards", type=float, default=0.0, help="randomly flips puzzle sides"
    )
    args = parser.parse_args()

    DEVICE = torch.device(args.device)

    if args.ieee_format:
        ieee_dataset = data_loader.load_ieee()
        eval_dataloader = DataLoader(ieee_dataset, batch_size=512)
    else:
        eval_dataset = data_loader.load_data(
            args.evaluation_corpus,
            flip_boards=args.flip_boards,
            use_cleaned_dataset=args.use_cleaned_datasets,
            use_maia2_features=args.use_maia2_features,
        )
        eval_dataloader = DataLoader(eval_dataset, batch_size=512)
    if args.ensemble_model:
        model = BatchedBaseline.deserialize(args.model_path)
    else:
        model = SimpleBaseline.deserialize(args.model_path)
    model = model.to(DEVICE)
    evaluate(
        model,
        eval_dataloader,
        device=DEVICE,
        verbose=args.verbose,
        ieee_format=args.ieee_format,
        ensemble_model=args.ensemble_model,
        rescale_dataloader=args.ensemble_model,
    )
