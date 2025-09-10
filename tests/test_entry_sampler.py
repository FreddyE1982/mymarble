import unittest
import sys
import pathlib
from collections import defaultdict

import torch

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
import main
from network.entities import Neuron
from network.graph import Graph
from network.entry_sampler import EntrySampler


class ReporterAdapter:
    def report(self, name, value):
        main.Reporter.report(name, f"Entry sampler metric {name}", value)


class TestEntrySampler(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        main.Reporter._metrics = {}
        self.zero = torch.tensor(0.0)
        self.reporter = ReporterAdapter()

    def test_probabilities_and_sampling_distribution(self):
        graph = Graph(reporter=main.Reporter)
        n1 = Neuron(zero=self.zero)
        n1.phi_v = torch.tensor(0.0)
        n2 = Neuron(zero=self.zero)
        n2.phi_v = torch.tensor(1.0)
        n3 = Neuron(zero=self.zero)
        n3.phi_v = torch.tensor(2.0)
        graph.add_neuron("a", n1)
        graph.add_neuron("b", n2)
        graph.add_neuron("c", n3)

        sampler = EntrySampler(
            temperature=torch.tensor(1.0),
            torch=torch,
            reporter=self.reporter,
        )

        probs = sampler.compute_probabilities(graph)
        expected = torch.softmax(torch.tensor([0.0, 1.0, 2.0]), dim=0)
        stacked = torch.stack([probs["a"], probs["b"], probs["c"]])
        for nid, prob in probs.items():
            main.Reporter.report(f"entry_prob_{nid}", f"Entry probability for {nid}", prob)
        print(
            "Probabilities:",
            [
                main.Reporter.report("entry_prob_a"),
                main.Reporter.report("entry_prob_b"),
                main.Reporter.report("entry_prob_c"),
            ],
        )
        self.assertTrue(torch.allclose(stacked, expected))
        self.assertAlmostEqual(float(stacked.sum().item()), 1.0, places=6)

        counts = defaultdict(int)
        samples = 5000
        for _ in range(samples):
            chosen = sampler.sample_entry()
            counts[chosen.id] += 1
        last_id = chosen.id
        freq = torch.tensor(
            [counts["a"], counts["b"], counts["c"]], dtype=torch.float32
        ) / samples
        print("Sampling frequencies:", freq.tolist())
        self.assertTrue(torch.allclose(freq, expected, atol=0.02))

        metric = main.Reporter.report("entry_sample")
        print("Last sampled neuron:", metric)
        self.assertEqual(metric, last_id)


if __name__ == "__main__":
    unittest.main()

