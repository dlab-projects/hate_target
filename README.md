# Targeted Identity Group Prediction in Hate Speech Corpora

by Pratik Sachdeva, Renata Barreto, Claudia von Vacano, and Chris J. Kennedy

## Overview 

This repository provides the code to reproduce the analyses and figures
developed in the paper [*Targeted Identity Group Prediction in Hate Speech
Corpora*](https://aclanthology.org/2022.woah-1.22/) by Sachdeva et al.,
published in 2022 at the 6th Workshop on Online Abuse and Harms (WOAH).

## Abstract

The past decade has seen an abundance of work seeking to detect, characterize,
and measure online hate speech. A related, but less studied problem, is the
detection of identity groups targeted by that hate speech. Predictive accuracy
on this task can supplement additional analyses beyond hate speech detection,
motivating its study. Using the Measuring Hate Speech corpus, which provided
annotations for targeted identity groups, we created neural network models to
perform multi-label binary prediction of identity groups targeted by a comment.
Specifically, we studied 8 broad identity groups and 12 identity sub-groups
within race and gender identity. We found that these networks exhibited good
predictive performance, achieving ROC AUCs of greater than 0.9 and PR AUCs of
greater than 0.7 on several identity groups. We validated their performance on
HateCheck and Gab Hate Corpora, finding that predictive performance generalized
in most settings. We additionally examined the performance of the model on
comments targeting multiple identity groups. Our results demonstrate the
feasibility of simultaneously identifying targeted groups in social media
comments.

## Repository Structure

The code used for this paper are divided into two repositories. To train the
target identity models, you will need the Tensorflow layers defined in the
[`hate_measure` repository](https://github.com/dlab-projects/hate_measure). To
run the scripts used to set up and train those models, run secondary analyses,
and generate the figures, you will need the code in this repository.

The repository is divided into the following folders:

* `figures`: Jupyter notebooks used to generate the figures in the paper.
* `hate_target`: Contains the codebase used in scripts, analyses, and figure
  generation for this paper.
* `notebooks`: Contains supplementary Jupyter notebooks used in secondary
  analyses.
* `scripts`: Contains Python scripts used to train variants of the model, whose
  predictions were analyzed in the paper.

## Set Up and Install

To run the code, first download this repository to your local machine. The
easiest way to do this is to clone to code via SSH:

```
git clone git@github.com:dlab-projects/hate_target.git
```

Navigate to the cloned folder on your local machine. Then, create a new Anaconda
environment with the `environment.yml` file:

```
conda env create -f environment.yml
```

Finally, install an editable version of this package using `pip`. Be sure to run
the following command in the `hate_target` folder, where `setup.py` is visible:

```
pip install -e .
```

You should now have access to `hate_target`'s functions as importable modules
anywhere in your virtual environment.

## Generate Figures

## Run the Code
