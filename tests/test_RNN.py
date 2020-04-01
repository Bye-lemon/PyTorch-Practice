from unittest import TestCase
from models.RNN import *

import numpy as np


class TestDictionary(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dic = Dictionary()
        cls.word_list = ["foo", "bar", "baz"]

    def test_add_word(self):
        try:
            for index, word in enumerate(self.word_list):
                assert index == self.dic.add_word(word)
        except AssertionError as e:
            print(e)
            self.fail()

    def test_onehot_encoded(self):
        try:
            onehot = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float)
            for index, word in enumerate(self.word_list):
                assert all(onehot[index] == self.dic.onehot_encoded(word))
        except AssertionError as e:
            print(e)
            self.fail()
