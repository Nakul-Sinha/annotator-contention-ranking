# Annotator Disagreement Ranking

## The problem

I get about 173 held out mouse face crops and have to rank them by how far an
expert panel's independent 0, 1 and 2 marks scatter. Only the ordering of my
priority score is scored, never the values themselves. The idea is triage: surface
the images where the assessors are least likely to agree.

## What I did

The first thing worth saying is that the label is a small sample statistic. The
exact contention values have denominators dividing 5 times k squared for roughly
3 to 5 assessors, which means the realized scatter carries something like 35
percent relative sampling noise. That rules out the obvious approach. A shrinking
L1 or L2 regression head is the wrong tool for a spread target, because it will
happily regress toward the mean and throw away exactly the ordering I am being
scored on.

The second problem is validation. No subject column is provided, and I verified
the ids are random rather than animal identifiers, so a naive random split leaks
the same animal across train and validation. I recover subject groups by Ward
clustering ResNet-18 embeddings of the training images into 141 clusters, which
gives sane group sizes and a positive target ICC, and I validate on those groups.

I also replicated the official metric locally and checked it against its anchors:
oracle scores 1.0000, random scores 0.000, reverse ordering and constant both
clip to 0.

## Layout

`solution.py` is the entry point and `code/metric.py` holds the metric replica.
`TECHNICAL.md` has the data profile, the validation design, and the experiment
log. Datasets are not committed.
