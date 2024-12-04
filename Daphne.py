import random

class Daphne:
    def __init__(self, vocab: set[str], random_seed: bool = True):
        self.vocab = vocab

        # for testing purposes
        if not random_seed:
            random.seed(0)

        self.unicode_list = self.get_unicode()
        
        # number of unicode characters that can be used per character so that each character is not represented by the same unicode character
        self.num_uni_per_char = int(len(self.unicode_list) / len(self.vocab))

        if self.num_uni_per_char < 1:
            raise ValueError("The character vocab is larger than the unicode list")
        
        # randomly shuffle the unicode list to remove patterns
        random.shuffle(self.unicode_list)

        self.map_unicode()

    def get_unicode(self) -> list:
        """
        Returns a list of all 6398 unicode characters in the Unicode Private Use Area
        """
        chars = []
        for hex in range(57344, 63742):
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
            for _ in range(self.num_uni_per_char - 1):
                unicode_val = next(unicode_iter)
                self.char_to_uni[char].append(unicode_val)
                self.uni_to_char[unicode_val] = char
    
    def encode(self, data: str) -> str:
        """
        Encodes the data using the character to unicode map
        """
        encoded_data = ""
        for char in data:
            if char not in self.char_to_uni.keys():
                raise ValueError("Character not in character vocab")

            encoded_data += self.char_to_uni[char][random.randint(0, self.num_uni_per_char - 1)]
        
        return encoded_data
    
    def decode(self, data: str) -> str:
        """
        Decodes the data using the unicode to character map
        """
        decoded_data = ""
        
        for char in data:
            decoded_data += self.uni_to_char[char]

        return decoded_data
