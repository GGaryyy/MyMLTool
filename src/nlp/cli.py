"""Command-line entry point for the Chinese-text NLP pipeline (Docker entrypoint).

Subcommands:
    diagnose         GPU / wheel-compatibility report (装机第一步验证)
    eda              run text EDA on a labelled CSV, write report to output_dir
    feature-select   analyse & recommend term / metadata features, write report
    benchmark        run the model benchmark, write report to output_dir
    download-models  pre-download compliant HF models for offline use

Heavy imports (torch/transformers) stay lazy: ``diagnose`` and ``download``
work even where the benchmark's deep-learning deps are absent.
"""

import argparse
import sys

from src.nlp.config import load_config


def _cmd_diagnose(args) -> int:
    from src.nlp.device import print_diagnostics

    print_diagnostics()
    return 0


def _cmd_eda(args) -> int:
    from src.nlp.datasets import load_text_dataset
    from src.nlp.eda import run_eda
    from src.nlp.report import write_eda_report

    config = load_config(args.config)
    dataset = load_text_dataset(config)

    # EDA covers the whole corpus; reunite splits and pass their row indices so
    # cross-split leakage is reported.
    texts = list(dataset.texts_train) + list(dataset.texts_val) + list(dataset.texts_test)
    raw_labels = _decode_labels(dataset, config.data.label_separator)
    splits = _split_index_map(dataset)

    report = run_eda(texts, raw_labels, config, splits=splits)
    written = write_eda_report(report, config.output_dir)
    print(f"EDA report written:\n  markdown: {written['markdown']}\n  json: {written['json']}")
    for warning in report.warnings:
        print(f"  [!] {warning}")
    return 0


def _cmd_feature_select(args) -> int:
    import pandas as pd

    from src.nlp.analysis.feature_selection import run_feature_selection
    from src.nlp.datasets import _resolve_column, _load_csv
    from src.nlp.report import write_feature_selection_report

    config = load_config(args.config)
    df = _load_csv(config.data.csv_path)
    text_name = _resolve_column(df, config.data.text_col, "text")
    label_name = _resolve_column(df, config.data.label_col, "label")
    if text_name == label_name:
        print(f"error: text_col and label_col resolve to the same column '{text_name}'",
              file=sys.stderr)
        return 2

    texts = [str(t) for t in df[text_name]]
    raw_labels = list(df[label_name])
    metadata_cols = [c for c in config.data.metadata_cols if c in df.columns]
    metadata_df = df[metadata_cols] if metadata_cols else None

    report = run_feature_selection(texts, raw_labels, config,
                                   metadata_df=metadata_df, out_dir=config.output_dir)
    written = write_feature_selection_report(report, config.output_dir)
    print("Feature-selection report written:")
    print(f"  markdown: {written['markdown']}")
    print(f"  json: {written['json']}")
    for plot in written["plots"]:
        print(f"  plot: {plot}")
    print("Top recommendations:")
    for rec in report.recommendations:
        print(f"  - {rec}")
    return 0


def _cmd_benchmark(args) -> int:
    from src.nlp.harness import run_benchmark
    from src.nlp.report import write_benchmark_report

    config = load_config(args.config)
    if not config.models:
        print("error: config has no 'models' to benchmark", file=sys.stderr)
        return 2

    result = run_benchmark(config)
    written = write_benchmark_report(result.to_dict(), config.output_dir)
    print(f"Benchmark report written:\n  markdown: {written['markdown']}\n  json: {written['json']}")
    print(f"Ranking: {' > '.join(result.ranking) if result.ranking else '(all models errored)'}")
    return 0


def _cmd_download_models(args) -> int:
    from scripts.download_models import main as download_main

    argv = ["--dest", args.dest]
    if args.models:
        argv += ["--models", *args.models]
    return download_main(argv)


def _decode_labels(dataset, separator: str) -> list:
    """Reconstruct raw labels from encoded splits for EDA class counting."""
    space = dataset.label_space
    import numpy as np

    y_all = np.concatenate([dataset.y_train, dataset.y_val, dataset.y_test], axis=0)
    decoded = space.decode(y_all)
    if space.is_multilabel:
        # EDA's build_label_space re-parses separator-joined strings.
        return [separator.join(labels) if labels else space.classes[0] for labels in decoded]
    return decoded


def _split_index_map(dataset) -> dict:
    """Row-index ranges for train/val/test in the reunited corpus order."""
    n_train = len(dataset.texts_train)
    n_val = len(dataset.texts_val)
    n_test = len(dataset.texts_test)
    splits = {"train": list(range(n_train))}
    if n_val:
        splits["val"] = list(range(n_train, n_train + n_val))
    splits["test"] = list(range(n_train + n_val, n_train + n_val + n_test))
    return splits


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.nlp.cli",
        description="中文文本 (繁中) NLP 分析與選型 pipeline",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_diag = sub.add_parser("diagnose", help="GPU / wheel 相容性診斷")
    p_diag.set_defaults(func=_cmd_diagnose)

    p_eda = sub.add_parser("eda", help="文本 EDA 資料分析")
    p_eda.add_argument("--config", required=True, help="EDA YAML 設定檔路徑")
    p_eda.set_defaults(func=_cmd_eda)

    p_fs = sub.add_parser("feature-select", help="特徵篩選分析與建議")
    p_fs.add_argument("--config", required=True, help="feature-select YAML 設定檔路徑")
    p_fs.set_defaults(func=_cmd_feature_select)

    p_bench = sub.add_parser("benchmark", help="演算法選型 benchmark")
    p_bench.add_argument("--config", required=True, help="benchmark YAML 設定檔路徑")
    p_bench.set_defaults(func=_cmd_benchmark)

    p_dl = sub.add_parser("download-models", help="預下載合規預訓練模型（離線用）")
    p_dl.add_argument("--dest", required=True, help="模型下載目標目錄 (HF_HOME)")
    p_dl.add_argument("--models", nargs="+", help="要下載的模型 key，預設 bert sent_embed")
    p_dl.set_defaults(func=_cmd_download_models)

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
