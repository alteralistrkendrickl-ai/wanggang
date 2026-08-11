import json
import os
import pickle
import tempfile
import unittest

import numpy as np

from convert_pkl_datasets import (
    _collect_class_records,
    _write_domain_protocol,
    _write_iid_protocol,
    convert_p3mc_strict,
)
from verify_manytx_protocol import verify_protocol, verify_strict_root


def _synthetic_tx(tx_index, rx_count=5, day_count=4, samples_per_domain=3):
    tx_data = []
    for rx_index in range(rx_count):
        rx_data = []
        for day_index in range(day_count):
            value = tx_index * 100 + rx_index * 10 + day_index
            block = np.full(
                (samples_per_domain, 256, 2),
                value,
                dtype=np.float32,
            )
            rx_data.append([block])
        tx_data.append(rx_data)
    return tx_data


class ManyTxProtocolTest(unittest.TestCase):
    def setUp(self):
        self.class_count = 3
        self.records = [
            _collect_class_records(_synthetic_tx(index), index, 0)
            for index in range(self.class_count)
        ]
        self.metadata = {
            "source": "synthetic.pkl",
            "equalized_index": 0,
            "tx_names": [f"tx-{index}" for index in range(self.class_count)],
            "rx_names": [f"rx-{index}" for index in range(5)],
            "day_names": [f"day-{index}" for index in range(4)],
        }

    def _load_domains(self, root, prefix, split):
        path = os.path.join(
            root,
            f"{prefix}_{split}_{self.class_count}Class.npy",
        )
        return set(np.load(path).astype(int).tolist())

    def test_cross_receiver_splits_have_no_receiver_leakage(self):
        with tempfile.TemporaryDirectory() as root:
            _write_domain_protocol(
                self.records,
                self.class_count,
                root,
                "rx",
                val_domains=[2, 3],
                test_domains=[4],
                metadata=self.metadata,
            )
            train = self._load_domains(root, "RX", "train")
            val = self._load_domains(root, "RX", "val")
            test = self._load_domains(root, "RX", "test")
            self.assertEqual(train, {0, 1})
            self.assertEqual(val, {2, 3})
            self.assertEqual(test, {4})
            self.assertFalse(train & val or train & test or val & test)
            report = verify_protocol(root, self.class_count)
            self.assertFalse(report["domain_leakage"])

    def test_cross_day_splits_have_no_day_leakage(self):
        with tempfile.TemporaryDirectory() as root:
            _write_domain_protocol(
                self.records,
                self.class_count,
                root,
                "day",
                val_domains=[2],
                test_domains=[3],
                metadata=self.metadata,
            )
            train = self._load_domains(root, "DAY", "train")
            val = self._load_domains(root, "DAY", "val")
            test = self._load_domains(root, "DAY", "test")
            self.assertEqual(train, {0, 1})
            self.assertEqual(val, {2})
            self.assertEqual(test, {3})
            self.assertFalse(train & val or train & test or val & test)
            report = verify_protocol(root, self.class_count)
            self.assertFalse(report["domain_leakage"])

    def test_iid_protocol_writes_fixed_validation_and_metadata(self):
        with tempfile.TemporaryDirectory() as root:
            summary = _write_iid_protocol(
                self.records,
                self.class_count,
                root,
                seed=2024,
                test_ratio=0.2,
                val_ratio=0.2,
                metadata=self.metadata,
            )
            self.assertTrue(all(summary[split]["samples"] > 0 for split in ("train", "val", "test")))
            with open(os.path.join(root, "protocol.json"), encoding="utf-8") as file:
                manifest = json.load(file)
            self.assertEqual(manifest["protocol"], "iid")
            self.assertIn(str(self.class_count), manifest["classes"])

    def test_strict_protocol_keeps_identity_and_domain_splits_disjoint(self):
        with tempfile.TemporaryDirectory() as root:
            dataset = {
                "data": [
                    _synthetic_tx(
                        index,
                        rx_count=3,
                        day_count=3,
                        samples_per_domain=70,
                    )
                    for index in range(9)
                ],
                "tx_list": [f"tx-{index}" for index in range(9)],
                "rx_list": [f"rx-{index}" for index in range(3)],
                "capture_date_list": [f"day-{index}" for index in range(3)],
                "equalized_list": [0],
            }
            pickle_path = os.path.join(root, "synthetic.pkl")
            with open(pickle_path, "wb") as file:
                pickle.dump(dataset, file)
            output_root = os.path.join(root, "converted")
            convert_p3mc_strict(
                pickle_path,
                output_root,
                protocols=("cross_rx", "cross_day"),
                val_rx=(1,),
                test_rx=(2,),
                val_day=(1,),
                test_day=(2,),
                base_indices=range(0, 4),
                validation_indices=range(4, 6),
                test_indices=range(6, 9),
                target_class_counts=(1, 2, 3),
            )

            with open(
                os.path.join(output_root, "strict_p3mc_protocol.json"),
                encoding="utf-8",
            ) as file:
                manifest = json.load(file)
            groups = manifest["identity_groups"]
            self.assertFalse(set(groups["base_pretraining"]) & set(groups["test_novel"]))
            self.assertEqual(manifest["target_tasks"]["3"], [6, 7, 8])

            cross_rx = os.path.join(output_root, "ManyTx_cross_rx")
            self.assertEqual(
                set(np.load(os.path.join(cross_rx, "TX_GLOBAL_train_4Class.npy"))),
                {0, 1, 2, 3},
            )
            self.assertEqual(
                set(np.load(os.path.join(cross_rx, "TX_GLOBAL_test_3Class.npy"))),
                {6, 7, 8},
            )
            self.assertEqual(len(np.load(os.path.join(cross_rx, "Y_test_3Class.npy"))), 150)
            report = verify_protocol(cross_rx, 3)
            self.assertFalse(report["domain_leakage"])
            strict_report = verify_strict_root(output_root)
            self.assertFalse(strict_report["identity_leakage"])


if __name__ == "__main__":
    unittest.main()
