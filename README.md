# Structured Embeddings

This repository contains code for analyzing how word usage differs across related groups of text data.
It contains 3 models you can fit to grouped text data:

 - global Bernoulli embeddings 
   (no variations between groups)
 - hierarchical Bernoulli embeddings
   (embeddings share statistical strength through hierarchical prior)
 - amortized Bernoulli embeddings
   (embeddings share statistical strength through amortization)
   
We use these models to analyze how the language of Senators differs according to their home state and party affiliation and how scientific language varies in differerent sections of the ArXiv.

The corresponding publication is

[M. Rudolph, F. Ruiz, S. Athey, D. Blei, **Structured Embedding Models for Grouped Data**, 
*Neural Information Processing Systems*, 2017](https://nips.cc/Conferences/2017/AcceptedPapersInitial)

Note: rightnow the link above links to NIPS' list of accepted papers, but we will upload a pdf version of our paper soon.
