from Daphne import Daphne

vocab = set("the quick brown fox jumps over the lazy dog., THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG!")
daphne = Daphne(vocab)

encoded = daphne.encode("Hello world!")
print("encoded:", encoded)

decoded = daphne.decode(encoded)
print("decoded:", decoded)