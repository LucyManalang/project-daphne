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
    parser.add_argument("--generate", action="store_true", 
                        help="Generate text using the transformer")
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
    
elif args.generate:
    for p in args.p.split(","):
        p = float(p)
        encoder = FileEncoder()

        if args.method == "seq":
            train_data = encoder.encode_seq(p)
            
        elif args.method == "words":
            train_data = encoder.encode_words(p)

        elif args.method == "letters":
            train_data = encoder.encode_letters(p)
        
        transformer = Transformer(train_data, seed, generate=True, prompt="The")
        print(f"Percentage: {p}")
        print(f"Generated text: {transformer.gen_text}")
        
elif args.method:
    for p in args.p.split(","):
        p = float(p)
        encoder = FileEncoder()
        valid_data = FileEncoder("brown-valid.txt").raw_data

        if args.method == "seq":
            train_data = encoder.encode_seq(p)
            
        elif args.method == "words":
            train_data = encoder.encode_words(p)

        elif args.method == "letters":
            train_data = encoder.encode_letters(p)

        transformer = Transformer(train_data, seed, valid_data=valid_data)

        print(f"Percentage: {p}")
        print(f"Perplexity: {transformer.perplexity}")
