import sys
import unittest
from unittest import mock

from utils.config import pretrain_config


class A1RunTagTest(unittest.TestCase):
    def make_config(self, *extra):
        argv = [
            "pretext.py", "--dataset", "wisig-cross-rx", "--TSLA_len", "256",
            "--TSLA_patch", "32", *extra,
        ]
        with mock.patch.object(sys, "argv", argv):
            return pretrain_config()

    def test_tag_is_appended_after_a1_variant(self):
        a1s = self.make_config("--a1_sampler", "--run_tag", "fair10e")
        a1c = self.make_config("--a1_enable", "--run_tag", "fair10e")
        self.assertTrue(a1s["exp_name"].endswith("_A1S_fair10e"))
        self.assertTrue(a1c["exp_name"].endswith("_A1C_fair10e"))

    def test_empty_tag_preserves_existing_paths(self):
        config = self.make_config("--a1_enable")
        self.assertTrue(config["exp_name"].endswith("_A1C"))
        self.assertNotIn("fair10e", config["exp_name"])

    def test_unsafe_tag_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "run_tag"):
            self.make_config("--a1_enable", "--run_tag", "../overwrite")


if __name__ == "__main__":
    unittest.main()
