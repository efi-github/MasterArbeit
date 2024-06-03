import unittest

import torch

from _0_mamba_vs_neo.models.MambaForSequenceClassification import MambaForSequenceClassification
from transformers import MambaConfig


class MambaForSequenceClassificationTest(unittest.TestCase):
    def test_init(self):
        # test it generally works
        try:
            MambaForSequenceClassification(MambaConfig())
        except Exception as e:
            self.fail(f"Failed to instantiate MambaForSequenceClassification: {e}")
        print("MambaForSequenceClassification instantiated successfully")

    def test_classifier_init(self):
        # there was the problem that the classifier initialized with zeros
        try:
            obj = MambaForSequenceClassification(MambaConfig())
            assert sum([abs(x) for x in obj.classifier.weight.flatten()]) > 0.5
        except Exception as e:
            self.fail(f"Failed to initialize classifier: {e}")
        print("Classifier initialized successfully")

    def test_forward(self):
        try:
            obj = MambaForSequenceClassification(MambaConfig())
            obj.pad_token_id = 0
            input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
            outputs = obj(input_ids)
            assert outputs.logits.shape == (2, 2)
        except Exception as e:
            self.fail(f"Failed to forward pass: {e}")
        print("Forward pass successful")

    def test_last_non_pad_positions(self):
        try:
            obj = MambaForSequenceClassification(MambaConfig())
            obj.pad_token_id = 0
            # generate random batch size
            batch_size = torch.randint(1, 100, (1,))
            # for each batch size generate random sequence length [0, 10)
            sequence_length = torch.randint(0, 10, (int(batch_size[0]),))
            # generate random input_ids
            input_ids = torch.randint(10, 1000, (int(batch_size[0]), 10))
            # set every element after sequence_length to be 0
            for i in range(int(batch_size[0])):
                input_ids[i, sequence_length[i] + 1:] = 0
            print(input_ids)
            res = obj._find_last_non_pad_position(input_ids)
            print(res)
            assert torch.equal(res, sequence_length)
        except Exception as e:
            self.fail(f"Failed to calculate last_non_pad_positions: {e}")
        print("last_non_pad_positions calculated successfully")

    def test_last_gpt2_non_pad_positions(self):
        try:
            obj = MambaForSequenceClassification(MambaConfig())
            obj.pad_token_id = 0
            # generate random batch size
            batch_size = torch.randint(1, 100, (1,))
            # for each batch size generate random sequence length [0, 10)
            sequence_length = torch.randint(0, 10, (int(batch_size[0]),))
            # generate random input_ids
            input_ids = torch.randint(10, 1000, (int(batch_size[0]), 10))
            # set every element after sequence_length to be 0
            for i in range(int(batch_size[0])):
                input_ids[i, sequence_length[i] + 1:] = 0
            print(input_ids)
            res = obj.gpt2_find_last_non_pad_position(input_ids)
            print(res)
            assert torch.equal(res, sequence_length)
        except Exception as e:
            self.fail(f"Failed to calculate last_non_pad_positions: {e}")
        print("last_non_pad_positions calculated successfully")

    def testResult(self):
        try:
            obj = MambaForSequenceClassification(MambaConfig())
            obj.pad_token_id = 0
            input_seq_1 = torch.tensor([[1, 2, 3, 4, 5]])
            input_seq_2 = torch.tensor([[1, 2, 3, 4, 5, 0]])
            res_1 = obj.forward(input_seq_1)
            res_2 = obj.forward(input_seq_2)
            assert torch.allclose(res_1.logits, res_2.logits, atol=1e-6, rtol=1e-4)
        except Exception as e:
            self.fail(f"Failed to correctly pass on correct last_non_pad_position: {e}\n{res_1.logits}\n{res_2.logits}")
        print("last_non_pad_positions correctly passed on the same result")


if __name__ == '__main__':
    unittest.main()
