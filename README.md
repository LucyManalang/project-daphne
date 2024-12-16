# project-daphne

Protection from data scrapers using unicode PUA. Inspired by [UChicago's Glaze/Nightshade](https://nightshade.cs.uchicago.edu/whatis.html)

# To run:

Daphne works by encoding the data using unicode PUA. For this project, I tried encoding the data in 3 different ways:

1. Encode each sequence of characters. In an article, this method would encode the full article. Different p (percentage) values can be used to encode a different percentage of sequences in the training data. This would replicate a scraper ending up with p percent of the data being encoded.

2. Encode each word. This was one of my two attempts to poison the training data.

3. Encode each letter. This was my other, more successful, attempt to poison the training data.

## Demo:

    python Main.py --demo

## Perplexity:

    python Main.py --method [seq|words|letters] --p [comma-separated-percentages]

## Generate:

    python Main.py --method --generate [seq|words|letters] --p [comma-separated-percentages]
