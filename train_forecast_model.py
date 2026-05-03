from __future__ import annotations

import argparse

from forecast_model import train_forecast_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the baseline traffic forecasting model from historical training data."
    )
    parser.add_argument(
        "--training-data",
        default="history/forecast_training_data.csv",
        help="CSV file containing supervised forecasting rows.",
    )
    parser.add_argument(
        "--model-output",
        default="models/traffic_forecast_model.json",
        help="Where to save the trained model bundle.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Ridge regularization strength for the baseline regression model.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    bundle = train_forecast_model(
        training_path=args.training_data,
        model_output_path=args.model_output,
        alpha=args.alpha,
    )
    print("Traffic forecast model trained")
    print(f"model_path: {bundle.model_path}")
    print(f"sample_count: {bundle.sample_count}")
    for metric, value in bundle.metrics.items():
        print(f"{metric}: {value:.4f}")


if __name__ == "__main__":
    main()
