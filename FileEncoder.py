from Daphne import Daphne

class FileEncoder:
    def __init__(self):
        with open("brown-train.txt") as file:
            self.raw_data = file.read()
            self.split_data = self.raw_data.split("<eos>")
        self.daphne = Daphne(set(self.raw_data))
    
    def encode_seq(self, p: float) -> list[str]:
        return self.daphne.encode_full_seq(self.split_data, p)
    
    def encode_words(self, p: float) -> list[str]:
        return self.daphne.encode_words(self.split_data, p)
    
    def encode_letters(self, p: float) -> list[str]:
        return self.daphne.encode_letters(self.split_data, p)
    
    def get_valid_data(self) -> list[str]:
        with open("brown-valid.txt") as file:
            split_data = file.read().split("<eos>")
        return split_data