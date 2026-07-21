"""Pre-download Hugging Face models for offline Chinese-text NLP deployment.

Run this in an internet-connected environment BEFORE `docker save`, so the
delivered image / mounted /models volume works with HF_HUB_OFFLINE=1 on the
air-gapped customer machine.

Only license-compliant, non-China-origin models are listed here (see
docs/nlp/LICENSES.md). CKIP (GPL-3.0) is intentionally excluded.

Usage:
    python scripts/download_models.py --dest /models
    python scripts/download_models.py --dest /models --models bert sent_embed
"""

import argparse
import sys

# name -> (repo_id, license, origin) ; kept in sync with docs/nlp/LICENSES.md
COMPLIANT_MODELS = {
    "bert": ("google-bert/bert-base-chinese", "Apache-2.0", "Google/US"),
    "xlmr": ("FacebookAI/xlm-roberta-base", "MIT", "Meta/US"),
    "sent_embed": (
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "Apache-2.0",
        "UKP-Lab/DE",
    ),
}
DEFAULT_MODELS = ("bert", "sent_embed")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Pre-download compliant HF models for offline use.")
    parser.add_argument("--dest", required=True, help="Target directory (becomes HF_HOME / mounted /models).")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=sorted(COMPLIANT_MODELS),
        default=list(DEFAULT_MODELS),
        help=f"Which models to fetch. Default: {' '.join(DEFAULT_MODELS)}",
    )
    return parser.parse_args(argv)


def download_one(key: str, dest: str) -> None:
    """Download tokenizer + weights for one registered model into ``dest``."""
    from huggingface_hub import snapshot_download

    repo_id, license_name, origin = COMPLIANT_MODELS[key]
    print(f"[download] {key}: {repo_id} ({license_name}, {origin})")
    snapshot_download(repo_id=repo_id, cache_dir=dest)
    print(f"[ok] {key} -> {dest}")


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        from huggingface_hub import snapshot_download  # noqa: F401
    except ImportError:
        print(
            "huggingface_hub is required. Install NLP deps first: "
            "pip install -r requirements-nlp.txt",
            file=sys.stderr,
        )
        return 1

    for key in args.models:
        download_one(key, args.dest)

    print(
        f"\nDone. On the offline machine set HF_HOME={args.dest} and "
        f"HF_HUB_OFFLINE=1, then reference the model by its repo id."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
