import random
import argparse

from Daphne import Daphne
from Transformer import Transformer
from FileEncoder import FileEncoder

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--demo", action="store_true", 
                        help="Demonstrate encodings")
    parser.add_argument("--method", type=str, 
                        help="Method to use for encoding")
    parser.add_argument("--p", type=str, default="0.5",
                        help="Percentages to use for encoding, comma seperated")
    parser.add_argument("--seed", type=int,
                        help="Seed for rng")
args = parser.parse_args()

if args.seed:
    seed = args.seed
else:
    seed = random.randint(0, 100)

if args.demo:
    vocab = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!.,;:?1234567890 ")
    daphne = Daphne(vocab, seed)

    text = ["the quick", "brown fox", "jumps over", "the lazy dog"]
    p = float(args.p.split(" ")[0])

    print("text:", text)
    print("p:", p)
    print("seed:", seed, "\n")

    encoded = daphne.encode_full_seq(text, p)
    print("encoded_full_seq: ", "['", "', '".join(encoded), "']", sep="")
    decoded = daphne.decode(encoded)
    print("decoded_full_seq:", decoded, "\n")

    encoded = daphne.encode_words(text, p)
    print("encoded_words: ", "['", "', '".join(encoded), "']", sep="")
    decoded = daphne.decode(encoded)
    print("decoded_words:", decoded, "\n")

    encoded = daphne.encode_letters(text, p)
    print("encoded_letters: ", "['", "', '".join(encoded), "']", sep="")
    decoded = daphne.decode(encoded)
    print("decoded_letters:", decoded, "\n")
    
elif args.method:
    for p in args.p.split(","):
        p = float(p)
        encoder = FileEncoder()
        valid_data = encoder.get_valid_data()

        if args.method == "seq":
            encoded = encoder.encode_seq(p)
            
        elif args.method == "words":
            encoded = encoder.encode_words(p)

        elif args.method == "letters":
            encoded = encoder.encode_letters(p)

        transformer = Transformer(encoded, valid_data, seed)

        print(f"Percentage: {p}")
        print(f"Perplexity: {transformer.perplexity}")
