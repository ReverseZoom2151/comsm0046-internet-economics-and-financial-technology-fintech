# Findings

What the three strands of this coursework actually show, measured rather than
asserted.

The originals were four Jupyter notebooks. Between 68% and 94% of each file was
stored cell output. One of them could not run at all, two of them were the same
notebook twice, and the most interesting exercises in the third were left
unfinished. This document records what they produce once that is fixed.

Everything here is reproducible:

```bash
python -m experiments.smith1962          # market simulation, about 2m40s
python -m experiments.week5_hypothesis   # hypothesis testing
python -m experiments.week7_sentiment    # sentiment analysis
```

## What was wrong

**The market notebook could not run.** It called `market_session` with eight
arguments. BSE 1.91 takes seven, and replaced the open file handle and the
`dump_all` boolean with a dictionary of flags. `BSE.py` was also absent from the
repository entirely, along with the diagram its markdown embeds. It is now
vendored, unmodified, with its MIT licence, and a test asserts its checksum so
an accidental edit fails the suite.

**Two of the four notebooks were the same notebook.** `week5_activity_dataset1`
and `week5_activity_dataset2` were 24 of 27 cells byte-identical, differing only
in which `read_csv` line was commented out. 272KB and 207KB for a one line
difference.

**The statistics ran a test whose assumption the preceding cell had rejected.**
Both notebooks ran Shapiro-Wilk and then ran one-way ANOVA regardless of the
answer, because nothing branched on the result.

**Three exercises were never done**, and they are the ones that carry the
lesson.

Smaller, and all fixed: the market notebook seeded one of six runs, built
arrays with `np.append` inside a per-row loop, left a file handle open, and
wrote session CSVs into whatever directory it was run from. The sentiment
notebook installed packages from a code cell, and its corpus had been through a
bad encoding round trip so pound signs and apostrophes were replacement
characters.

## Market simulation: Vernon Smith's 1962 experiment

Bristol Stock Exchange, seed 100, 10 sessions per scenario, 600 simulated
seconds in 10 trading periods. Convergence is Smith's coefficient alpha, the
root mean squared deviation of transaction prices from the equilibrium price,
as a percentage of it. Lower is more converged.

| Scenario | Trades | Equilibrium | Alpha first | Alpha last | Silent periods |
|---|---|---|---|---|---|
| 11 ZIP, periodic 60s | 245 | 200.0 | 20.54 | 18.12 | 1 of 10 |
| 11 ZIP, drip-poisson 10s | 281 | 200.0 | 14.18 | 4.50 | 5 of 10 |
| 40 ZIP, drip-poisson 10s | 745 | 199.5 | 8.99 | 11.08 | 7 of 10 |
| Mixed ZIP/ZIC/SHVR/GVWY, 80 traders | 11650 | 199.5 | 7.56 | 4.73 | 0 of 10 |
| Mixed, with a mid-session shock | 11470 | 199.5 | 7.56 | 1.41 | 0 of 10 |

### Smith's finding replicates, where the market keeps trading

In the mixed population alpha falls from 7.56 to 4.87 within a single period and
holds near 4.7 for the rest of the session, at a mean transaction price of
199.77 against an equilibrium of 199.5. That is Smith's result: prices converge
on the competitive equilibrium quickly, without anybody being told what it is.

**The market shock is the cleanest demonstration.** The equilibrium jumps from
199.5 to 349.5 half way through:

| Period | Equilibrium | Mean price | Alpha |
|---|---|---|---|
| 5 | 199.5 | 199.86 | 4.73 |
| 6 | 349.5 | 330.06 | 12.68 |
| 7 | 349.5 | 347.47 | 1.27 |
| 8 | 349.5 | 347.15 | 1.36 |
| 9 | 349.5 | 347.14 | 1.32 |
| 10 | 349.5 | 346.65 | 1.41 |

Alpha spikes in the period containing the shock and then settles to around 1.3,
which is **tighter than the market ever tracked the original equilibrium**. The
population re-converges inside one trading period on a price nobody told it.

### The homogeneous markets converge and then stop

The all-ZIP markets look worse on the headline numbers, and the reason is not
that they fail to converge. They are silent for 5 of 10 periods at 11 traders a
side and 7 of 10 at 40. They converge, run out of profitable trades, and lock
up. The apparent late divergence in the baseline, alpha rising from 8.4 to
18.12, is 14 trades in the final period, which is noise rather than a result.

The original notebook mentions the lock-up in prose. Counting the silent periods
turns that remark into a number, which is why the summary reports the trade
count of the final period alongside alpha.

### Two defects found by measuring rather than reading

**The notebook read cancellations as prices.** It took column 2 of every tape
row as a transaction price, but BSE writes cancellations as `CAN` rows where
that column holds an order id. Filtering on the event type is now tested.

**The equilibrium is not 200.** BSE truncates each schedule step with `int()`,
so with 40 traders a side the price ladder is asymmetric and the market clears
at 199.5. Scoring convergence against a hard-coded 200 would have reported a
bias that is an artefact of the assignment code rather than a property of the
market.

## Hypothesis testing

| | data1 | data2 |
|---|---|---|
| Conditions | 3 (a, b, c) | 2 (x, y) |
| n per condition | 15 | 20 |
| Shapiro-Wilk | p = 0.984, 0.675, 0.725 | **p = 0.0118, 0.0303** |
| Holm corrected | 1.000 throughout | 0.0236, 0.0303 |
| Normal? | yes | **no, both reject** |
| Levene equal variance | p = 0.5935, yes | p = 0.6827, yes |
| Test chosen | one-way ANOVA | Kruskal-Wallis |
| Result | F = 2.565, p = 0.0889 | H = 9.196, p = 0.00243 |
| Significant at 0.05 | no | yes |

**Dataset 2 fails its normality assumption and the notebooks ran ANOVA on it
anyway.** Both conditions reject Shapiro-Wilk, and both survive a Holm
correction. The notebook computed that result, printed it, asked the reader to
think about it, and then ran a parametric test regardless, because no code
branched on the answer.

The conclusion survives: Kruskal-Wallis gives p = 0.00243 against the ANOVA's
p = 0.00281, so both say the two conditions differ. The answer was right and the
route to it was not justified. That is worth separating, because the same fixed
pipeline applied to different data is exactly how an unjustified method goes
unnoticed, and here the pipeline was applied twice because the notebook had been
duplicated.

Three further gaps: Levene's test for equal variance was never run at all, so
ANOVA's second assumption went unchecked; there was no post-hoc, so for
dataset 1's three conditions the omnibus test could not have said which differ;
and `sns.barplot(ci='sd')` used a parameter seaborn 0.13 has removed, under a
title calling a standard deviation a confidence interval.

Worth stating plainly: Shapiro-Wilk at n = 15 has low power, so failing to
reject normality for dataset 1 is close to a foregone conclusion rather than
evidence of normality.

## Sentiment analysis, and why it should not drive a trading signal

The three unfinished exercises are the point of the notebook, and the measured
answers are worse than the lesson implies.

### The Associated Press hack, 23 April 2013

> "Breaking: Two Explosions in the White House and Barack Obama is injured"

**Polarity 0.0000. Subjectivity 0.0000.**

Not merely wrong. TextBlob scores *none* of the words in that sentence. Not
"explosions", not "injured", not "breaking". Its lexicon is built from film
reviews and contains none of them. To this analyser a false report of an attack
on the head of state, which moved US equity indices by about one percent in
three minutes, is indistinguishable from an empty string.

The subjectivity of 0.0000 matters as much as the polarity. The obvious defence,
acting only on scores the model is confident about, would rate this tweet
maximally factual.

### Muddy Waters, 6 August 2019

> "Muddy Waters is now in a blackout period until tomorrow 8 am London time when
> we will announce a new short position on an accounting fiasco that's
> potentially insolvent and possibly facing a liquidity crunch..."

**Polarity +0.0341. Subjectivity 0.6886.**

Mildly *positive*, the evening before the target lost more than half its market
value. The only word scored at all is "new", at +0.14. "Short position",
"accounting fiasco", "insolvent" and "liquidity crunch" all score zero, and
"We're not", the clause that reverses the entire message, scores zero as a two
word fragment.

### What that means

Both failures are the dangerous kind. The first gives no signal before a large
move, so a polarity threshold stays flat through the event and then trades the
reversal at the worst available price. The second gives a signal of the wrong
sign.

A general-purpose lexicon has no finance vocabulary, no negation handling across
clause boundaries, and no notion of who is speaking or whether the claim is
true. The aspect-level machinery in the rest of the notebook is a genuine
improvement on document-level scoring for product reviews, where it lines up
with the review's own summary: battery +0.227 and size +0.226 positive, power
-0.044 negative. None of that helps on either tweet, because the problem is the
lexicon, not the granularity.

### An index-space bug in the aspect splitting

Aspect positions were recorded as word indices and then used to slice a list of
n-gram polarities. A sentence of W words has only W-n+1 n-grams, so every
interior cut landed (n-1)//2 positions late, and the final boundary overshot the
list silently, because Python clips slices. The plot then labelled each point
with a word it did not cover, and the last two labels had no curve beneath them.

With the mapping corrected, the trough at -0.5 sits under "bit expensive given"
and the rise at +0.7 under "processor seems really good", which is what the
sentence says.

## Reproducing all of this

```bash
pip install -r requirements-dev.txt

python -m experiments.smith1962
python -m experiments.week5_hypothesis
python -m experiments.week7_sentiment --download   # first run only, fetches the tokenizer
```
