import unittest
import sys
import pathlib
import torch

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
from main import Reporter
from network.entities import Neuron, Synapse
from network.graph import Graph


class TestGraphForward(unittest.TestCase):
    def test_forward_pipeline(self):
        torch.manual_seed(0)
        zero = torch.tensor(0.0)
        n1 = Neuron(zero=zero)
        n2 = Neuron(zero=zero)
        n1.phi_v = torch.tensor(1.0)
        n2.phi_v = torch.tensor(-10.0)
        n1.last_local_loss = torch.tensor(0.1)
        n1.lambda_v = torch.tensor(0.2)
        n2.last_local_loss = torch.tensor(0.0)
        n2.lambda_v = torch.tensor(0.3)
        s1 = Synapse(zero=zero)
        s1.c_e = torch.tensor(0.2)
        s1.lambda_e = torch.tensor(0.4)
        Reporter._metrics = {}
        g = Graph(reporter=Reporter)
        g.add_neuron("n1", n1)
        g.add_neuron("n2", n2)
        g.add_synapse("s1", "n1", "n2", s1)
        result = g.forward(
            method="exact",
            cost_params={"lambda_0": 0, "lambda_max": 1, "alpha": 1, "beta": 1, "T_heat": 1},
        )
        path = result["path"]
        self.assertEqual(path[0], n1)
        self.assertEqual(path[1], s1)
        self.assertEqual(path[2], n2)
        print("Returned path_time:", result.get("path_time"))
        print("Returned final loss:", result.get("final_cumulative_loss"))
        self.assertIn("path_time", result)
        self.assertIn("final_cumulative_loss", result)
        self.assertEqual(Reporter.report("entry_id"), "n1")
        self.assertEqual(Reporter.report("path_id"), 0)
        self.assertIsNotNone(Reporter.report("path_cost"))
        self.assertIsNotNone(Reporter.report("path_time"))
        self.assertIsNotNone(Reporter.report("final_cumulative_loss"))

    def test_forward_with_scalar_threshold(self):
        torch.manual_seed(0)
        zero = 0
        n1 = Neuron(zero=zero)
        n2 = Neuron(zero=zero)
        n1.phi_v = torch.tensor(1.0)
        n2.phi_v = torch.tensor(-10.0)
        n1.last_local_loss = torch.tensor(0.1)
        n1.lambda_v = torch.tensor(0.2)
        n2.last_local_loss = torch.tensor(0.0)
        n2.lambda_v = torch.tensor(0.3)
        s1 = Synapse(zero=zero)
        s1.c_e = torch.tensor(0.2)
        s1.lambda_e = torch.tensor(0.4)
        Reporter._metrics = {}
        g = Graph(reporter=Reporter)
        g.add_neuron("n1", n1)
        g.add_neuron("n2", n2)
        g.add_synapse("s1", "n1", "n2", s1)
        result = g.forward(
            method="exact",
            cost_params={"lambda_0": 0, "lambda_max": 1, "alpha": 1, "beta": 1, "T_heat": 1},
        )
        path = result["path"]
        self.assertEqual(path[0], n1)
        self.assertEqual(path[1], s1)
        self.assertEqual(path[2], n2)
        print("Scalar threshold path_time:", result.get("path_time"))
        print("Scalar threshold final loss:", result.get("final_cumulative_loss"))
        self.assertIn("path_time", result)
        self.assertIn("final_cumulative_loss", result)

    def test_forward_uses_weights(self):
        torch.manual_seed(0)
        zero = torch.tensor(0.0)
        Reporter._metrics = {}
        n1 = Neuron(zero=zero, reporter=Reporter)
        n2 = Neuron(zero=zero, reporter=Reporter)
        n1.phi_v = torch.tensor(1.0)
        n2.phi_v = torch.tensor(-10.0)
        n1.last_local_loss = torch.tensor(1.0)
        n2.last_local_loss = torch.tensor(1.0)
        n1.update_weight(torch.tensor(2.0))
        n2.update_weight(torch.tensor(3.0))
        s1 = Synapse(zero=zero, reporter=Reporter)
        s1.update_weight(torch.tensor(4.0))
        g = Graph(reporter=Reporter)
        g.add_neuron("n1", n1)
        g.add_neuron("n2", n2)
        g.add_synapse("s1", "n1", "n2", s1)
        result = g.forward(method="exact")
        print("Weighted forward cumulative n1:", n1.cumulative_loss)
        print("Weighted forward cumulative n2:", n2.cumulative_loss)
        self.assertAlmostEqual(n1.cumulative_loss.item(), 2.0)
        self.assertAlmostEqual(n2.cumulative_loss.item(), 12.0)
        self.assertEqual(result["final_cumulative_loss"], n2.cumulative_loss)
        self.assertEqual(
            Reporter.report(f"neuron_{id(n1)}_weight").item(), 2.0
        )
        self.assertEqual(
            Reporter.report(f"synapse_{id(s1)}_weight").item(), 4.0
        )


if __name__ == "__main__":
    unittest.main()
