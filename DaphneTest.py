import unittest
from Daphne import Daphne

# python3 -m unittest DaphneTest.py
class TestDaphne(unittest.TestCase):
    def setUp(self):
        self.vocab = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!.,;:?1234567890 ")
        self.daphne = Daphne(self.vocab, seed=0)
        self.len = 6398

    def test_correct_num_uni_per_char(self):
        self.assertEqual(self.daphne.num_uni_per_char, int(self.len / len(self.vocab)) - 1)
    
    def test_get_unicode(self):
        unicode_list = self.daphne.get_unicode()
        self.assertEqual(len(unicode_list), self.len)

        all_in_pua = True
        for uni in unicode_list:
            if ord(uni) not in range(57344, 63742):
                all_in_pua = False
                break
        self.assertTrue(all_in_pua)
    
    def test_map_unicode(self):
        self.assertEqual(len(self.daphne.char_to_uni), len(self.vocab))
        self.assertEqual(len(self.daphne.uni_to_char), (self.daphne.num_uni_per_char) * len(self.daphne.vocab))

    def test_encode_full_seq(self):
        text = ["the quick", "brown fox", "jumps over", "the lazy dog"]
        encoded = self.daphne.encode_full_seq(text, 0.5)
        decoded = self.daphne.decode(encoded)
        self.assertEqual(decoded, text)
    
    def test_encode_words(self):
        text = ["the quick", "brown fox", "jumps over", "the lazy dog"]
        encoded = self.daphne.encode_words(text, 0.5)
        decoded = self.daphne.decode(encoded)
        self.assertEqual(decoded, text)

    def test_encode_letters(self):
        text = ["the quick", "brown fox", "jumps over", "the lazy dog"]
        encoded = self.daphne.encode_letters(text, 0.5)
        decoded = self.daphne.decode(encoded)
        self.assertEqual(decoded, text)
