import random

class Daphne:
    def __init__(self, vocab: set[str], seed: int = 0):
        self.vocab = vocab

        random.seed(seed)

        self.unicode_list = self.get_unicode()
        
        # number of unicode characters that can be used per character so that each character is not represented by the same unicode character
        self.num_uni_per_char = int(len(self.unicode_list) / len(self.vocab)) - 1

        if self.num_uni_per_char < 1:
            raise ValueError("The character vocab is larger than the unicode list")
        
        # randomly shuffle the unicode list to remove patterns
        random.shuffle(self.unicode_list)

        self.map_unicode()

    def get_unicode(self) -> list:
        """
        Returns a list of all 65,534 unicode characters in the Unicode Private Use Area A
        """
        chars = []
        for hex in range(983040, 1048573):
            uni = chr(hex)
            chars.append(uni)
        return chars

    def map_unicode(self):
        """
        Maps each character to a list of random unicode characters for encoding
        Maps each unicode character to its corresponding character for decoding
        """
        self.char_to_uni = {}
        self.uni_to_char = {}

        unicode_iter = iter(self.unicode_list)
        
        for char in self.vocab:
            unicode_val = self.unicode_list.pop()

            self.char_to_uni[char] = []
            for _ in range(self.num_uni_per_char):
                unicode_val = next(unicode_iter)
                self.char_to_uni[char].append(unicode_val)
                self.uni_to_char[unicode_val] = char
    
    def encode_full_seq(self, data: list[str], p: float) -> list[str]:
        """
        Encodes p percent of the data by encoding the entire sequence
        """
        encoded_data = []
        for seq in data:
            if random.random() > p:
                encoded_data.append(seq)
            else:
                encoded_seq = ""
                for char in seq:
                    if char not in self.char_to_uni.keys():
                        raise ValueError(f"Character '{char}' not in character vocab")

                    encoded_seq += self.char_to_uni[char][random.randint(0, self.num_uni_per_char - 1)]
                encoded_data.append(encoded_seq)
        return encoded_data
    
    def encode_words(self, data: list[str], p: float) -> list[str]:
        """
        Encodes p percent of the data by encoding words in the sequence
        """
        encoded_data = []
        for seq in data:
            encoded_seq = ""
            for word in seq.split(" "):
                if random.random() > p:
                    encoded_seq += word
                else:
                    for char in word:
                        if char not in self.char_to_uni.keys():
                            raise ValueError(f"Character '{char}' not in character vocab")

                        encoded_seq += self.char_to_uni[char][random.randint(0, self.num_uni_per_char - 1)]
                encoded_seq += " "
            encoded_data.append(encoded_seq[:-1])
        return encoded_data
    
    def encode_letters(self, data: list[str], p: float) -> list[str]:
        """
        Encodes p percent of the data by encoding letters in the sequence
        """
        encoded_data = []
        for seq in data:
            encoded_seq = ""
            for char in seq:
                if random.random() > p:
                    encoded_seq += char
                else:
                    if char not in self.char_to_uni.keys():
                        raise ValueError(f"Character '{char}' not in character vocab")

                    encoded_seq += self.char_to_uni[char][random.randint(0, self.num_uni_per_char - 1)]
            encoded_data.append(encoded_seq)
        return encoded_data
    
    def decode(self, data: list[str]) -> list[str]:
        """
        Decodes the data using the unicode to character map
        """
        decoded_data = []
        
        for seq in data:
            decoded_seq = ""
            for char in seq:
                decoded_seq += self.uni_to_char.get(char, char)
            decoded_data.append(decoded_seq)

        return decoded_data
