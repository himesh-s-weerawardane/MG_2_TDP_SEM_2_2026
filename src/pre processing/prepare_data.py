import html
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from datasets import load_from_disk

TWEET_EVAL_PATH = Path("data/raw/tweet_eval_sentiment")
STOCK_EMOTIONS_PATH = Path("data/raw/StockEmotions/tweet")
OUTPUT_DIR = Path("data/processed")
REPORT_DIR = Path("reports")
SPLITS = ["train", "validation", "test"]
STOCK_FILES = ["train_stockemo.csv", "val_stockemo.csv", "test_stockemo.csv"]


def clean_text(value):
    if pd.isna(value):
        return ""
    text = html.unescape(str(value))
    text = re.sub(r"(?:https?://|www\.)\S+", " [URL] ", text, flags=re.I)
    text = re.sub(r"(?<!\w)@\w+", " [USER] ", text)
    return re.sub(r"\s+", " ", text).strip()


def prepare_frame(frame, domain, split):
    frame = frame.copy()
    before = len(frame)
    missing = int(frame["text"].isna().sum())
    frame["text"] = frame["text"].apply(clean_text)
    empty = int((frame["text"] == "").sum())
    frame = frame[frame["text"] != ""].copy()
    duplicates = int(frame.duplicated(["text", "label"]).sum())
    frame = frame.drop_duplicates(["text", "label"]).reset_index(drop=True)
    conflicting = int((frame.groupby("text")["label"].nunique() > 1).sum())

    if not frame["label"].isin([0, 1]).all():
        raise ValueError(f"Invalid labels in {domain} {split}.")
    frame["label"] = frame["label"].astype(int)
    if domain == "general":
        frame["emotion"] = frame["label"].map({0: "negative", 1: "positive"})
    frame["domain"] = domain
    frame["source_split"] = split
    frame["tokens"] = frame["text"].apply(str.split)
    frame["word_count"] = frame["tokens"].apply(len)
    frame["character_count"] = frame["text"].str.len()

    stats = {
        "rows_before_cleaning": before,
        "missing_text": missing,
        "empty_after_cleaning": empty,
        "duplicates_removed": duplicates,
        "rows_after_cleaning": len(frame),
        "conflicting_label_texts": conflicting,
    }

    print(f"\n{domain.upper()} - {split}")
    print(f"Rows before cleaning: {before}")
    print(f"Missing texts: {missing}; empty after cleaning: {empty}; duplicates removed: {duplicates}")
    print(f"Rows after cleaning: {len(frame)}")
    print(f"0 (negative/bearish): {(frame['label'] == 0).sum()}")
    print(f"1 (positive/bullish): {(frame['label'] == 1).sum()}")
    print(f"Average word count: {frame['word_count'].mean():.1f}")
    return frame, stats


def save_csv(frame, filename):
    saved = frame.copy()
    saved["tokens"] = saved["tokens"].apply(lambda tokens: json.dumps(tokens, ensure_ascii=False))
    saved.to_csv(OUTPUT_DIR / filename, index=False)
    print(f"Saved: {OUTPUT_DIR / filename}")


def create_eda(source, target, mixed, cleaning_stats):
    statistics = []
    overlaps = []
    for name, frames in [("TweetEval", source), ("StockEmotions", target)]:
        for split, frame in frames.items():
            row = {"dataset": name, "split": split}
            row.update(cleaning_stats[(name, split)])
            row.update({
                "label_0": int((frame["label"] == 0).sum()),
                "label_1": int((frame["label"] == 1).sum()),
                "mean_words": round(float(frame["word_count"].mean()), 2) if len(frame) else 0,
                "mean_characters": round(float(frame["character_count"].mean()), 2) if len(frame) else 0,
            })
            statistics.append(row)
        # Audit the supplied splits without silently changing their membership.
        for first, second in [("train", "validation"), ("train", "test"), ("validation", "test")]:
            count = len(set(frames[first]["text"]) & set(frames[second]["text"]))
            overlaps.append(f"{name}: {first}/{second} shared texts = {count}")
    for split in ["validation", "test"]:
        count = len(set(mixed["text"]) & set(target[split]["text"]))
        overlaps.append(f"Mixed train/finance {split} shared texts = {count}")

    summary = pd.DataFrame(statistics)
    summary.to_csv(REPORT_DIR / "data_summary.csv", index=False)
    report = (
        "PREPROCESSING AND DATA ANALYSIS\n\n"
        "Labels: 0 = negative/bearish; 1 = positive/bullish.\n"
        "Raw TweetEval neutral rows are excluded to align both datasets.\n"
        "Cleaning: decode HTML, replace URLs/mentions, normalize whitespace,\n"
        "remove empty text and duplicate text/label rows within each split.\n"
        "Preserve case, punctuation, emojis, hashtags, tickers and negation.\n"
        "Tokenization: whitespace tokens for EDA; use the selected BERT\n"
        "tokenizer during training and fit TF-IDF on training text only.\n"
        "Mixed training contains only general train + finance train.\n"
        "Finance validation and test remain separate for model evaluation.\n\n"
        + summary.to_string(index=False)
        + "\n\nSplit overlap audit:\n" + "\n".join(overlaps)
        + "\nReview nonzero overlaps before training to avoid data leakage.\n"
        "Class imbalance and domain-specific sentiment meanings should be\n"
        "considered in the report; labels may not reflect every interpretation.\n"
    )
    (REPORT_DIR / "data_analysis_report.txt").write_text(report, encoding="utf-8")

    counts = pd.DataFrame({
        "General": source["train"]["label"].value_counts().reindex([0, 1], fill_value=0),
        "Finance": target["train"]["label"].value_counts().reindex([0, 1], fill_value=0),
    }).T
    counts.columns = ["0: negative/bearish", "1: positive/bullish"]
    counts.plot.bar(rot=0, figsize=(8, 5))
    plt.title("Training sentiment distribution")
    plt.ylabel("Number of texts")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "class_distribution.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    for name, frame in [("General", source["train"]), ("Finance", target["train"])]:
        plt.hist(frame["word_count"], bins=30, alpha=0.5, density=True, label=name)
    plt.title("Training text length distribution")
    plt.xlabel("Whitespace tokens per text")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "text_length_distribution.png", dpi=150)
    plt.close()
    print("\nEDA statistics:")
    print(summary.to_string(index=False))
    print("\n" + "\n".join(overlaps))
    print(f"EDA report and charts saved to {REPORT_DIR}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    print("Starting preprocessing: 0 = negative/bearish; 1 = positive/bullish")
    tweet_eval = load_from_disk(str(TWEET_EVAL_PATH))
    training_frames = []
    source = {}
    target = {}
    cleaning_stats = {}

    for split, stock_file in zip(SPLITS, STOCK_FILES):
        general = tweet_eval[split].to_pandas()[["text", "label"]].copy()
        if not general["label"].isin([0, 1, 2]).all():
            raise ValueError(f"Unexpected TweetEval labels in {split}.")
        print(f"\nTweetEval {split}: removing {(general['label'] == 1).sum()} neutral rows.")
        general = general[general["label"] != 1].copy()
        general["label"] = general["label"].map({0: 0, 2: 1})

        raw = pd.read_csv(STOCK_EMOTIONS_PATH / stock_file)
        finance = pd.DataFrame({
            "text": raw["original"],
            "label": raw["senti_label"].str.strip().str.lower().map({"bearish": 0, "bullish": 1}),
            "ticker": raw["ticker"],
            "emotion": raw["emo_label"],
        })

        general, general_stats = prepare_frame(general, "general", split)
        finance, finance_stats = prepare_frame(finance, "finance", split)
        source[split] = general
        target[split] = finance
        cleaning_stats[("TweetEval", split)] = general_stats
        cleaning_stats[("StockEmotions", split)] = finance_stats
        save_csv(general, f"tweet_eval_{split}.csv")
        save_csv(finance, f"stock_emotions_{split}.csv")
        if split == "train":
            training_frames = [general, finance]

    mixed = pd.concat(training_frames, ignore_index=True)
    save_csv(mixed, "mixed_train.csv")
    print(f"\nMixed train: {len(mixed)} rows")
    print(mixed.groupby(["domain", "label"]).size().to_string())
    create_eda(source, target, mixed, cleaning_stats)
    print("\nPreprocessing completed successfully.")


if __name__ == "__main__":
    main()