import unittest
import torch
import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
import main
from network.entities import Neuron, Synapse
from network.graph import Graph
from network.complexity import ComplexityCalculator
from network.topology_fitness import TopologyFitness
from network.evolution import EvolutionOperator
from routing.improvements import GateAdjuster


class DeterministicSampler:
    def __init__(self, target_id):
        self._target = target_id
        self.graph = None

    def compute_probabilities(self, graph):
        self.graph = graph
        for nid, neuron in graph.neurons.items():
            setattr(neuron, "id", nid)
        return {nid: (1.0 if nid == self._target else 0.0) for nid in graph.neurons}

    def sample_entry(self):
        return self.graph.neurons[self._target]


class TestEvolutionaryStep(unittest.TestCase):
    def setUp(self):
        main.Reporter._metrics = {}
        torch.manual_seed(0)
        self.zero = torch.tensor(0.0)
        self.A = [torch.tensor(0.5)]
        self.B = [torch.tensor(0.3)]
        self.gamma = torch.tensor(0.1)
        self.lambda_ = torch.tensor(0.2)
        self.attr_sets = [{"x", "y"}]
        self.learning_rate = torch.tensor(0.1)

    def _build_graph(self):
        weight = torch.tensor([1.0, -2.0], requires_grad=True)
        graph = Graph(reporter=main.Reporter)
        n1 = Neuron(zero=self.zero, activation_threshold=torch.tensor(0.5, requires_grad=True))
        n2 = Neuron(zero=self.zero, activation_threshold=torch.tensor(0.5, requires_grad=True))
        graph.add_neuron("n1", n1)
        graph.add_neuron("n2", n2)
        s1 = Synapse(zero=self.zero)
        graph.add_synapse("s1", "n1", "n2", s1)
        n1.record_local_loss(torch.tensor(0.5))
        n2.record_local_loss(torch.tensor(0.4))
        n1.update_latency(torch.tensor(0.1))
        n2.update_latency(torch.tensor(0.1))
        s1.c_e = torch.tensor(0.05)
        s1.lambda_e = torch.tensor(0.05)
        complexity = ComplexityCalculator(
            self.A, self.B, self.gamma, self.lambda_, [weight], self.attr_sets, reporter=main.Reporter
        )
        topology = TopologyFitness(reporter=main.Reporter)
        graph._complexity_calculator = complexity
        graph._topology_fitness = topology
        graph._entry_sampler = DeterministicSampler("n1")
        evo = EvolutionOperator(
            graph,
            complexity,
            topology,
            torch.tensor(0.1),
            torch.tensor(0.1),
            torch.tensor(0.1),
            torch.tensor(0.1),
            torch.tensor(0.1),
            reporter=main.Reporter,
        )
        graph._evolution_operator = evo
        graph._routing_adjuster = GateAdjuster(reporter=main.Reporter, learning_rate=self.learning_rate)
        return graph, n1, n2, s1, weight

    def _expected_complexity(self, num_neurons, num_edges, weight):
        l1 = weight.abs().sum()
        l2 = weight.pow(2).sum().sqrt()
        base = self.zero + num_neurons
        contrib = (
            self.A[0] * num_neurons
            + self.B[0] * num_edges
            + self.gamma * l1
            + self.lambda_ * l2
        )
        return base + contrib + len(self.attr_sets[0])

    def test_pipeline_complexity_fitness_and_gate(self):
        graph, n1, n2, s1, weight = self._build_graph()
        ctx = {
            "neuron_id": "n_new",
            "neuron": Neuron(zero=self.zero),
            "source_id": "n1",
            "target_id": "n2",
            "synapse_in_id": "s_in",
            "synapse_in": Synapse(zero=self.zero),
            "synapse_out_id": "s_out",
            "synapse_out": Synapse(zero=self.zero),
            "b_vn": torch.tensor(1.0),
            "g_vn": torch.tensor(0.5),
            "latency": torch.tensor(0.1),
            "cost": torch.tensor(0.1),
        }
        result = graph.forward(global_loss_target=self.zero, evolution_instructions={"add_neuron": ctx})
        cost = main.Reporter.report("path_cost")
        latency = torch.tensor(0.25)
        final_loss = result["final_cumulative_loss"]
        expected_pre_c = self._expected_complexity(2, 1, weight)
        expected_post_c = self._expected_complexity(3, 3, weight)
        expected_path_val = -(final_loss + latency + cost)
        expected_pre_f = expected_path_val - expected_pre_c
        expected_post_f = expected_path_val - expected_post_c
        pre_c = main.Reporter.report("complexity_before")
        post_c = main.Reporter.report("complexity_after")
        pre_f = main.Reporter.report("fitness_before")
        post_f = main.Reporter.report("fitness_after")
        print("Complexities:", pre_c, post_c)
        print("Fitness values:", pre_f, post_f)
        self.assertTrue(torch.allclose(pre_c, expected_pre_c))
        self.assertTrue(torch.allclose(post_c, expected_post_c))
        self.assertTrue(torch.allclose(pre_f, expected_pre_f))
        self.assertTrue(torch.allclose(post_f, expected_post_f))
        per_cost = cost / 2
        expected_thresh = torch.tensor(0.5) + self.learning_rate * (n1.lambda_v + per_cost)
        print("Threshold n1:", n1.activation_threshold)
        self.assertTrue(torch.allclose(n1.activation_threshold, expected_thresh))
        self.assertTrue(torch.allclose(n2.activation_threshold, expected_thresh))

    def test_mutation_thresholds_respected(self):
        graph, n1, n2, s1, weight = self._build_graph()
        add_ctx = {
            "neuron_id": "n_fail",
            "neuron": Neuron(zero=self.zero),
            "source_id": "n1",
            "target_id": "n2",
            "synapse_in_id": "s_in_f",
            "synapse_in": Synapse(zero=self.zero),
            "synapse_out_id": "s_out_f",
            "synapse_out": Synapse(zero=self.zero),
            "b_vn": torch.tensor(0.05),
            "g_vn": torch.tensor(0.02),
            "latency": torch.tensor(0.1),
            "cost": torch.tensor(0.1),
        }
        rem_metrics = {
            "b_vn": torch.tensor(0.05),
            "g_vn": torch.tensor(0.02),
            "latency": torch.tensor(0.1),
            "cost": torch.tensor(0.1),
        }
        add_syn_ctx = {
            "synapse_id": "s2",
            "source_id": "n1",
            "target_id": "n2",
            "synapse": Synapse(zero=self.zero),
            "b_e": torch.tensor(0.05),
            "g_e": torch.tensor(0.02),
            "latency": torch.tensor(0.1),
            "cost": torch.tensor(0.1),
        }
        rem_syn_metrics = {
            "b_e": torch.tensor(0.05),
            "g_e": torch.tensor(0.02),
            "latency": torch.tensor(0.1),
            "cost": torch.tensor(0.1),
        }
        move_metrics = rem_syn_metrics
        instructions = {
            "add_neuron": add_ctx,
            "remove_neuron": ("n2", rem_metrics),
            "add_synapse": add_syn_ctx,
            "remove_synapse": ("s1", rem_syn_metrics),
            "move_synapse": ("s1", "n2", "n1", move_metrics),
        }
        result = graph.forward(global_loss_target=self.zero, evolution_instructions=instructions)
        mutations = result.get("mutations", {})
        print("Mutation results:", mutations)
        self.assertFalse(mutations.get("add_neuron"))
        self.assertFalse(mutations.get("remove_neuron"))
        self.assertFalse(mutations.get("add_synapse"))
        self.assertFalse(mutations.get("remove_synapse"))
        self.assertFalse(mutations.get("move_synapse"))
        self.assertEqual(len(graph.neurons), 2)
        self.assertEqual(len(graph.synapses), 1)
        src, tgt, _ = graph.synapses["s1"]
        self.assertEqual(src, "n1")
        self.assertEqual(tgt, "n2")
        self.assertNotIn("s2", graph.synapses)
        self.assertEqual(main.Reporter.report("add_neuron_applied"), 0)
        self.assertEqual(main.Reporter.report("remove_neuron_applied"), 0)
        self.assertEqual(main.Reporter.report("add_synapse_applied"), 0)
        self.assertEqual(main.Reporter.report("remove_synapse_applied"), 0)
        self.assertEqual(main.Reporter.report("move_synapse_applied"), 0)


if __name__ == "__main__":
    unittest.main()
