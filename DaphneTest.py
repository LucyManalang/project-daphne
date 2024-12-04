import unittest
from Daphne import Daphne

class TestDaphne(unittest.TestCase):
    def setUp(self):
        self.vocab = set("the quick brown fox jumps over the lazy dog THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG ,.:;!? 1234567890")
        self.daphne = Daphne(self.vocab, random_seed=False)

    def test_correct_num_uni_per_char(self):
        self.assertEqual(self.daphne.num_uni_per_char, int(6398 / len(self.vocab)))
    
    def test_get_unicode(self):
        unicode_list = self.daphne.get_unicode()
        self.assertEqual(len(unicode_list), 6398)

        all_in_pua = True
        for uni in unicode_list:
            if ord(uni) not in range(57344, 63742):
                all_in_pua = False
                break
        self.assertTrue(all_in_pua)
    
    def test_map_unicode(self):
        self.assertEqual(len(self.daphne.char_to_uni), len(self.vocab))
        self.assertEqual(len(self.daphne.uni_to_char), (self.daphne.num_uni_per_char - 1) * len(self.daphne.vocab))

    def test_encode_decode(self):
        encoded = self.daphne.encode("Hello world!")
        decoded = self.daphne.decode(encoded)
        self.assertEqual(decoded, "Hello world!")
