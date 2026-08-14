# Findings

What the three strands of this coursework actually show, measured rather than
asserted.

The originals were four Jupyter notebooks. Between 68% and 94% of each file was
stored cell output. One of them could not run at all, two of them were the same
notebook twice, and the most interesting exercises in the third were left
unfinished. This document records what they produce once that is fixed.

Everything here is reproducible:

```bash
python -m experiments.smith1962             # market simulation, about 3m
python -m experiments.strategy_profit       # profit by strategy, with tests
python -m experiments.week5_hypothesis      # hypothesis testing
python -m experiments.week7_sentiment       # sentiment analysis
python -m experiments.sentiment_evaluation  # the two analysers, scored
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

Alpha spikes in the period containing the shock and then settles to around 1.3.
The population re-converges inside one trading period on a price nobody told it,
which is the result worth having.

**An earlier version of this document went further and was wrong.** It said the
shocked market ended up tighter than it had ever tracked the original
equilibrium, reading the 1.41 in the last row against the 4.73 five rows above.
Tested over 20 sessions on per-session alpha, the difference runs the other way:
across the post-shock periods the shocked market averages 6.03 against the
unshocked mixed market's 4.99 on the same periods, Welch p = 1.9e-5. The 1.41
is a single period of one run, and it excludes the shock period itself, which
carries alpha 12.68 on its own. Re-converging quickly is real. Converging
*better* than before was an artefact of reading two point estimates off a
table.

### Two of the four convergence claims here do not survive testing

The scenarios above are compared with this repository's own hypothesis testing
pipeline, over per-session alpha, so each scenario contributes a sample of 20
rather than a single number. Holm corrected across the four claims:

| Claim | Verdict |
|---|---|
| The mixed population converges better than all-ZIP | **holds**, 5.52 against 8.62, p = 6.7e-5 |
| A thicker all-ZIP market converges better | **holds**, 8.62 against 11.57, p = 0.0018 |
| Drip-poisson converges better than periodic arrival | **fails**, 11.57 against 13.20, p = 0.16 |
| The shocked market ends tighter than it began | **contradicted**, and reversed, p = 1.9e-5 |

The drip-poisson failure is instructive. The table above shows alpha_last of
18.12 for periodic against 4.50 for drip-poisson, which looks decisive and is
not: those are 14 trades and **2 trades** respectively. Two point estimates off
the tails of nearly silent periods. Measured over whole sessions the two arrival
modes are indistinguishable.

Note also that the 40-a-side scenario reads as worse in the table, alpha_last
11.08 against 4.50, and better on per-session alpha, 8.62 against 11.57. Same
reason: the final-period figures are computed on a handful of trades.

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

### Which trading strategy makes the most money

The notebook never asks, and BSE records the answer: its balance dump carries
mean profit per trader by strategy. Mixed market, 40 sessions, 20 traders per
strategy per side, seed 100.

| Strategy | Mean profit per trader | sd | sem |
|---|---|---|---|
| SHVR | 1859 | 470 | 74 |
| GVWY | 1714 | 436 | 69 |
| ZIP | 1637 | 414 | 65 |
| ZIC | 1446 | 503 | 79 |

One-way ANOVA gives F = 5.673, p = 0.00103. Tukey HSD separates only two pairs:
SHVR above ZIC (p = 0.0005) and GVWY above ZIC (p = 0.046). Everything else is
indistinguishable, and **ZIP, the only strategy here that adapts, is not
significantly ahead of anything.** At 20 sessions the same comparison gave
p = 0.057, so the sample size is doing real work and is quoted alongside.

Head to head, 40 sessions per pair, Holm corrected across six pairs: SHVR beats
ZIC (p = 7.7e-7) and GVWY (9.5e-5), and ZIC beats GVWY (1.9e-8). ZIP wins
nothing and loses nothing.

**Profit is a property of the pairing, not of the strategy**, and the data
demonstrates it rather than merely warning about it: ZIC comes last in the mixed
market and beats GVWY decisively one against one, while GVWY comes second in
that same mixed market. The ordering is not transitive, so neither result is
quotable without the other.

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

### Measured, not just illustrated

Two tweets are an anecdote. To turn the claim into a measurement, 96 labelled
headlines were written covering profit warnings, earnings beats, short seller
reports, dividend cuts, upgrades, insolvency language and neutral procedural
announcements, including the hard cases deliberately: negation, and words that
change sign between general English and finance. They are author written rather
than a published benchmark, which bounds what they can show, and the labels were
fixed before any analyser was run.

| | TextBlob | Finance lexicon |
|---|---|---|
| Accuracy | **34.4%**, 95% CI [25.6%, 44.3%] | **66.7%**, 95% CI [56.8%, 75.3%] |
| Negative precision / recall | 0.400 / 0.182 | 0.889 / 0.727 |
| Neutral precision / recall | 0.306 / 0.594 | 0.534 / 0.969 |
| Positive precision / recall | 0.421 / 0.258 | 0.818 / 0.290 |

**The majority class baseline is 34.4%.** Answering "neutral" to every headline
scores exactly what TextBlob scores. On this set it carries no information at
all.

McNemar's test on the paired predictions, which is the right test when two
classifiers score the same items rather than two independent samples: 28 both
correct, 27 both wrong, 5 TextBlob only, 36 lexicon only, chi-square 21.95,
p = 2.8e-06.

The honest summary is narrower than "the finance lexicon wins". It is much
better at recognising bad news, recall 0.727 against 0.182, and **no better than
the neutral guess at recognising good news**, recall 0.290. Loughran-McDonald's
positive list is deliberately short, because filings state good news plainly,
and it contains almost none of the vocabulary of a good headline: beat, raised,
record, upgrade, premium.

TextBlob's seven wrong-sign errors are the instructive ones. "Full year earnings
are **not** expected to meet guidance" scores positive. "**Free** cash flow
turned negative and net debt rose to four times earnings" scores positive. "The
loss **narrowed** sharply and management now expects to break even" scores
negative.

The finance lexicon corrects the Muddy Waters tweet to -1.0, on "insolvent". It
still scores the Associated Press tweet 0.0 with no word matched: a finance
dictionary has no more vocabulary for explosions and injuries than a film review
one does.

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
python -m experiments.sentiment_evaluation
```

The profit table above is the one figure here that a bare command does not
reproduce. It needs the published sample size explicitly:

```bash
python -m experiments.strategy_profit --sessions 40 --each 20
```

That takes over fifteen minutes, because the head to head arm runs six pairs of
strategies over forty sessions each. The default of twenty sessions runs in a
couple of minutes and **does not reach the same conclusion**: the profit
comparison gives p = 0.057 there against p = 0.00103 at forty. That difference
is the point about sample size rather than a caveat to it, so both are quoted.
