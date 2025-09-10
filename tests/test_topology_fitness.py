import unittest
import torch
import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
import main
from network.topology_fitness import TopologyFitness


class TestTopologyFitness(unittest.TestCase):
    def setUp(self):
        main.Reporter._metrics = {}

    def test_evaluate_topology_fitness(self):
        paths_stats = {
            "src1": [
                {
                    "loss": torch.tensor(0.5, requires_grad=True),
                    "latency": torch.tensor(0.1, requires_grad=True),
                    "cost": torch.tensor(0.2, requires_grad=True),
                },
                {
                    "loss": torch.tensor(0.3, requires_grad=True),
                    "latency": torch.tensor(0.2, requires_grad=True),
                    "cost": torch.tensor(0.1, requires_grad=True),
                },
            ],
            "src2": [
                {
                    "loss": torch.tensor(0.4, requires_grad=True),
                    "latency": torch.tensor(0.3, requires_grad=True),
                    "cost": torch.tensor(0.2, requires_grad=True),
                }
            ],
        }
        complexity = torch.tensor(0.6, requires_grad=True)
        tf = TopologyFitness(main.Reporter)
        fitness = tf.evaluate(paths_stats, complexity)
        expected = -(
            torch.tensor(0.5 + 0.1 + 0.2)
            + torch.tensor(0.3 + 0.2 + 0.1)
            + torch.tensor(0.4 + 0.3 + 0.2)
        ) - complexity
        self.assertTrue(torch.allclose(fitness, expected))
        fitness.backward()
        self.assertIsNotNone(complexity.grad)
        self.assertIsNotNone(paths_stats["src1"][0]["loss"].grad)
        self.assertIsNotNone(paths_stats["src1"][1]["latency"].grad)
        self.assertIsNotNone(paths_stats["src2"][0]["cost"].grad)
        self.assertIsNotNone(main.Reporter.report("fitness_value"))
        self.assertIsNotNone(main.Reporter.report("total_path_loss"))
        self.assertIsNotNone(main.Reporter.report("total_path_latency"))
        self.assertIsNotNone(main.Reporter.report("total_path_cost"))


if __name__ == "__main__":
    unittest.main()
