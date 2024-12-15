# >>> _, _, test_dataset = data_loader.load_data()
# >>> model = SimpleBaseline.deserialize("experiment_models/example_test/example_test.best.ckpt")
# >>> model = model.to(DEVICE)
# >>> test_dataloader = DataLoader(test_dataset, batch_size=256)
# >>> evaluate(model, test_dataloader, device=DEVICE)
import torch
from ensemble_baseline import *
from regression_baseline import *


def evaluate(
    model,
    eval_data,
    device=torch.device(0),
    verbose=False,
    ieee_format=False,
    ensemble_model=False,
):
    model.eval()

    preds_list, ratings_list = [], []
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
        themes,
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
        themes = themes.to(device)
        ratings = ratings.to(device)
        if ensemble_model:
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
        else:
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
                themes,
                num_moves,
            )

        preds_list.extend(pred.reshape(-1).tolist())
        ratings_list.extend(ratings.reshape(-1).tolist())
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
            themes,
            num_moves,
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
        )
        eval_dataloader = DataLoader(eval_dataset, batch_size=512)
    if args.ensemble_model:
        model = EnsembleBaseline.deserialize(args.model_path)
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
    )
