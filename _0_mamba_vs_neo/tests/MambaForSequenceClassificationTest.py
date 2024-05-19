import unittest
from _0_mamba_vs_neo.models.MambaForSequenceClassification import MambaForSequenceClassification
from transformers import MambaConfig


class MambaForSequenceClassificationTest(unittest.TestCase):
    def test_init(self):
        # test it generally works
        try:
            MambaForSequenceClassification(MambaConfig())
        except Exception as e:
            self.fail(f"Failed to instantiate MambaForSequenceClassification: {e}")

        # test that the classification head is not initialized to all zeros
        # TODO

    def test_forward(self):
        # test it generally works
        # TODO

        # check last_non_pad_token is calculated correctly for a diverse batch
        # TODO
        pass


if __name__ == '__main__':
    unittest.main()
