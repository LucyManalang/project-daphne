from Daphne import Daphne

class FileEncoder:
    def __init__(self, file_path: str = "brown-train.txt"):
        with open(file_path) as file:
            self.raw_data = file.read()
            self.split_data = self.raw_data.split("<eos>")
        self.daphne = Daphne(set(self.raw_data))
    
    def encode_seq(self, p: float) -> list[str]:
        return self.daphne.encode_full_seq(self.split_data, p)
    
    def encode_words(self, p: float) -> list[str]:
        return self.daphne.encode_words(self.split_data, p)
    
    def encode_letters(self, p: float) -> list[str]:
        return self.daphne.encode_letters(self.split_data, p)