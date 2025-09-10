import unittest
import sys
import pathlib
from collections import defaultdict

import torch

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
import main
from network.entities import Neuron
from network.path_selector import PathSelector


class TestPathSelection(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        main.Reporter._metrics = {}
        self.zero = torch.tensor(0.0)

    def test_select_exact(self):
        neuron = Neuron(zero=self.zero)
        paths = [
            (("p0",), torch.tensor(3.0)),
            (("p1",), torch.tensor(1.0)),
            (("p2",), torch.tensor(2.0)),
        ]
        selector = PathSelector(reporter=main.Reporter)
        best, sampled = selector.select_exact(neuron, paths)
        self.assertIs(best, paths[1][0])
        self.assertIs(sampled, paths[1][0])
        selected_metric = main.Reporter.report(f"neuron_{id(neuron)}_selected_path")
        sampled_cost = main.Reporter.report(f"neuron_{id(neuron)}_sampled_path_cost")
        print("Selected index:", selected_metric)
        print("Sampled path cost:", sampled_cost)
        self.assertEqual(selected_metric, 1)
        self.assertTrue(torch.allclose(sampled_cost, paths[1][1]))

    def test_select_soft_distribution(self):
        neuron = Neuron(zero=self.zero)
        paths = [
            (("p0",), torch.tensor(1.0)),
            (("p1",), torch.tensor(2.0)),
            (("p2",), torch.tensor(3.0)),
        ]
        selector = PathSelector(reporter=main.Reporter)
        counts = defaultdict(int)
        draws = 5000
        for _ in range(draws):
            best, sampled = selector.select_soft(
                neuron,
                paths,
                R_v_star=torch.tensor(0.0),
                T_sample=torch.tensor(1.0),
            )
            for idx, (seq, cost) in enumerate(paths):
                if sampled is seq:
                    counts[idx] += 1
                    last_idx = idx
                    break
        costs = torch.tensor([c.item() for _, c in paths])
        rewards = torch.exp((torch.tensor(0.0) - costs) / 1.0)
        expected = rewards / rewards.sum()
        freq = torch.tensor([counts[0], counts[1], counts[2]], dtype=torch.float32) / draws
        print("Sampling frequencies:", freq.tolist())
        self.assertTrue(torch.allclose(freq, expected, atol=0.02))
        selected_metric = main.Reporter.report(f"neuron_{id(neuron)}_selected_path")
        sampled_cost = main.Reporter.report(f"neuron_{id(neuron)}_sampled_path_cost")
        print("Selected index:", selected_metric)
        print("Sampled path cost:", sampled_cost)
        self.assertEqual(selected_metric, 0)
        self.assertTrue(torch.allclose(sampled_cost, paths[last_idx][1]))


if __name__ == "__main__":
    unittest.main()

