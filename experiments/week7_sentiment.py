"""Week 7: what TextBlob actually says about four texts, two of which moved markets.

Run with `python -m experiments.week7_sentiment`.

The reviews are the teaching material: they show that document level sentiment
hides the disagreement between sentences, and that sentence and aspect level
analysis recovers some of it. The two tweets are the reason the material matters
to a trading system. Both are texts that moved a market within minutes, and the
question this script answers by measurement rather than by assertion is whether
a polarity score would have told you anything useful about either one.

Nothing here asserts an expected number. It prints what the analyser says.

Add `--download` to fetch the NLTK tokenizer corpora if they are missing. That
is the only thing in this repository that touches the network, and it is opt in.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from fintech.plotting import save_figure
from fintech.reviews import (
    HACK_CRASH_TWEET,
    LONG_REVIEW,
    MAC_ASPECTS,
    MUDDY_WATERS_TWEET,
    SHORT_REVIEW,
)
from fintech.sentiment import (
    CorpusUnavailableError,
    analyse_text,
    aspect_sentiment_frame,
    ensure_corpora,
    mean_aspect_polarity,
    plot_sentence_sentiment,
    sentence_sentiment_frame,
)

FIGURES = Path(__file__).resolve().parent.parent / "figures"

RULE = "=" * 78


def _heading(title: str) -> None:
    print(f"\n{RULE}\n{title}\n{RULE}")


def _report_document(analysis) -> None:
    doc = analysis.document
    print(f"polarity     {doc.polarity:+.4f}   ({doc.label})")
    print(f"subjectivity {doc.subjectivity:.4f}")
    if analysis.notes:
        for note in analysis.notes:
            print(f"note: {note}")


def _report_sentences(analysis, limit: int = 6) -> None:
    df = analysis.sentences
    if df is None or df.empty:
        return
    ordered = df.sort_values(by="Polarity", ascending=False)
    print(f"\n{len(df)} sentences, most positive first (showing up to {limit} either end):")
    columns = ["Polarity", "Subjectivity", "Text"]
    head = ordered.head(limit)[columns]
    tail = ordered.tail(limit)[columns]
    for _, row in head.iterrows():
        print(f"  {row['Polarity']:+.3f}  {row['Subjectivity']:.2f}  {row['Text'][:90]}")
    if len(ordered) > 2 * limit:
        print("  ...")
        for _, row in tail.iterrows():
            print(f"  {row['Polarity']:+.3f}  {row['Subjectivity']:.2f}  {row['Text'][:90]}")


def _report_aspects(analysis) -> None:
    df = analysis.aspects
    if df is None or df.empty:
        return
    means = mean_aspect_polarity(df)
    print(f"\naspect level sentiment ({len(df)} mentions, {len(means)} distinct aspects):")
    for _, row in means.iterrows():
        print(f"  {row['Polarity']:+.3f}  {row['Aspect']:<12} ({int(row['Mentions'])} mentions)")


def analyse_review(name: str, text: str, title: str) -> None:
    _heading(title)
    analysis = analyse_text(text, name=name, aspects=MAC_ASPECTS)
    _report_document(analysis)
    _report_sentences(analysis)
    _report_aspects(analysis)


def analyse_tweet(name: str, text: str, title: str, context: str) -> None:
    _heading(title)
    print(f"{text}\n")
    print(f"context: {context}\n")
    analysis = analyse_text(text, name=name)
    _report_document(analysis)
    _report_sentences(analysis)
    if analysis.sentences is not None and not analysis.sentences.empty:
        # Word level, because it shows which tokens the analyser can see at all.
        words = [str(w) for s in analysis.sentences["Sentence"] for w in s.words]
        scored = [(w, analyse_text(w, name=w).document.polarity) for w in words]
        opinionated = [(w, p) for w, p in scored if p != 0.0]
        print("\nwords TextBlob scores at all:")
        print("  " + (", ".join(f"{w} {p:+.2f}" for w, p in opinionated) or "none"))


def write_aspect_figure(sentence_index: int, text: str, title: str) -> None:
    """Plot one multi-aspect sentence, which is where the index bug used to show."""

    df = sentence_sentiment_frame(text, MAC_ASPECTS)
    sentence = df.loc[sentence_index, "Sentence"]
    fig = plot_sentence_sentiment(sentence, MAC_ASPECTS)
    path = save_figure(fig, title, FIGURES)
    print(f"\nwrote {path}")


def write_aspect_bar_figure(text: str, title: str) -> None:
    """Mean polarity per aspect over the whole of the long review."""

    from fintech.plotting import new_figure

    means = mean_aspect_polarity(aspect_sentiment_frame(text, MAC_ASPECTS))
    if means.empty:
        return
    fig, ax = new_figure(width=9.0, height=4.5)
    colours = ["tab:green" if p >= 0 else "tab:red" for p in means["Polarity"]]
    ax.bar(means["Aspect"], means["Polarity"], color=colours)
    ax.axhline(0.0, color="grey", linewidth=0.8)
    ax.set_ylabel("mean polarity")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=90)
    path = save_figure(fig, title, FIGURES)
    print(f"wrote {path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--download",
        action="store_true",
        help="fetch the NLTK tokenizer corpora if they are missing (needs network access)",
    )
    args = parser.parse_args(argv)

    try:
        ensure_corpora(download=args.download)
    except CorpusUnavailableError as exc:
        print(f"warning: {exc}\nContinuing with document level analysis only.")

    analyse_review("short_review", SHORT_REVIEW, "1. Short MacBook review (teaching example)")
    analyse_review("long_review", LONG_REVIEW, "2. Long MacBook review (T3)")

    analyse_tweet(
        "hack_crash_tweet",
        HACK_CRASH_TWEET,
        "3. The hack crash tweet, Associated Press account, 23 April 2013",
        "false, posted by whoever had taken the account, and the S&P 500 lost "
        "around one per cent in the minutes that followed",
    )
    analyse_tweet(
        "muddy_waters_tweet",
        MUDDY_WATERS_TWEET,
        "4. Muddy Waters, 6 August 2019",
        "trailed a short position, announced the next morning against Burford "
        "Capital, whose shares then fell by more than half",
    )

    try:
        ensure_corpora()
    except CorpusUnavailableError:
        return 0

    write_aspect_figure(4, SHORT_REVIEW, "Aspect split of the price storage processor sentence")
    write_aspect_bar_figure(LONG_REVIEW, "Mean aspect polarity in the long MacBook review")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
