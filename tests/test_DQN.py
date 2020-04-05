from unittest import TestCase

from models.DQN import *


class TestReplyMemory(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.memory = ReplyMemory(5)

    def test_push(self):
        try:
            for i in range(6):
                self.memory.push(i, i, i, i)
            assert tuple(zip(*self.memory.memory))[0] == (5, 1, 2, 3, 4)
            assert tuple(zip(*self.memory.memory))[1] == (5, 1, 2, 3, 4)
            assert tuple(zip(*self.memory.memory))[2] == (5, 1, 2, 3, 4)
            assert tuple(zip(*self.memory.memory))[3] == (5, 1, 2, 3, 4)
        except AssertionError as e:
            print(e)
            self.fail()

    def test_sample(self):
        try:
            sample = self.memory.sample(2)
            assert len(sample) == 2
            assert sample[0] in self.memory.memory
            assert sample[1] in self.memory.memory
        except AssertionError as e:
            print(e)
            self.fail()


class TestDQNNet(TestCase):
    def test_forward(self):
        try:
            model = DQNNet()
            input_ = torch.randn(16, 3, 40, 80)
            output = model(input_)
            assert output.shape == (16, 2)
        except Exception as e:
            print(e)
            self.fail()
