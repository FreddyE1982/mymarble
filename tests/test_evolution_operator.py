import unittest
import torch
import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
import main
from network.graph import Graph
from network.entities import Neuron, Synapse
from network.complexity import ComplexityCalculator
from network.topology_fitness import TopologyFitness
from network.evolution import EvolutionOperator


class TestEvolutionOperator(unittest.TestCase):
    def setUp(self):
        main.Reporter._metrics = {}
        zero = torch.tensor(0.0)
        one = torch.tensor(1.0)
        self.zero = zero
        self.one = one
        self.graph = Graph(reporter=main.Reporter)
        n1 = Neuron(zero=zero)
        n2 = Neuron(zero=zero)
        self.graph.add_neuron('n1', n1)
        self.graph.add_neuron('n2', n2)
        s1 = Synapse(zero=zero)
        self.graph.add_synapse('s1', 'n1', 'n2', s1)
        weights = [torch.tensor([1.0], requires_grad=True)]
        A = [one]
        B = [one]
        gamma = zero
        lambda_ = zero
        self.complexity = ComplexityCalculator(A, B, gamma, lambda_, weights, reporter=main.Reporter)
        self.fitness = TopologyFitness(reporter=main.Reporter)
        self.paths_stats = {'n1': [{'loss': zero, 'latency': zero, 'cost': zero}]}
        self.operator = EvolutionOperator(
            self.graph,
            self.complexity,
            self.fitness,
            theta_add=torch.tensor(0.5),
            theta_remove=torch.tensor(0.5),
            theta_add_synapse=torch.tensor(0.5),
            theta_remove_synapse=torch.tensor(0.5),
            theta_move_synapse=torch.tensor(0.5),
            reporter=main.Reporter,
        )

    def test_add_and_remove_neuron(self):
        ctx = {
            'neuron_id': 'n_new',
            'neuron': Neuron(zero=self.zero),
            'source_id': 'n1',
            'target_id': 'n2',
            'synapse_in_id': 's_in',
            'synapse_in': Synapse(zero=self.zero),
            'synapse_out_id': 's_out',
            'synapse_out': Synapse(zero=self.zero),
            'b_vn': torch.tensor(1.0),
            'g_vn': torch.tensor(1.0),
            'latency': torch.tensor(0.1),
            'cost': torch.tensor(0.1),
        }
        applied = self.operator.add_neuron(ctx, self.paths_stats)
        self.assertTrue(applied)
        self.assertIn('n_new', self.graph.neurons)
        self.assertEqual(len(self.graph.synapses), 3)
        pre_c = main.Reporter.report('complexity_before')
        post_c = main.Reporter.report('complexity_after')
        self.assertTrue(torch.allclose(pre_c, torch.tensor(5.0)))
        self.assertTrue(torch.allclose(post_c, torch.tensor(9.0)))
        metrics = {
            'b_vn': torch.tensor(2.0),
            'g_vn': torch.tensor(0.2),
            'latency': torch.tensor(0.1),
            'cost': torch.tensor(0.1),
        }
        applied_rem = self.operator.remove_neuron('n_new', metrics, self.paths_stats)
        self.assertTrue(applied_rem)
        self.assertNotIn('n_new', self.graph.neurons)
        rem_pre = main.Reporter.report('complexity_before')
        rem_post = main.Reporter.report('complexity_after')
        self.assertTrue(torch.allclose(rem_pre, torch.tensor(9.0)))
        self.assertTrue(torch.allclose(rem_post, torch.tensor(5.0)))

    def test_synapse_operations(self):
        syn_ctx = {
            'synapse_id': 's2',
            'source_id': 'n1',
            'target_id': 'n2',
            'synapse': Synapse(zero=self.zero),
            'b_e': torch.tensor(1.0),
            'g_e': torch.tensor(1.0),
            'latency': torch.tensor(0.1),
            'cost': torch.tensor(0.1),
        }
        applied = self.operator.add_synapse(syn_ctx, self.paths_stats)
        self.assertTrue(applied)
        self.assertIn('s2', self.graph.synapses)
        pre_c = main.Reporter.report('complexity_before')
        post_c = main.Reporter.report('complexity_after')
        self.assertTrue(torch.allclose(pre_c, torch.tensor(5.0)))
        self.assertTrue(torch.allclose(post_c, torch.tensor(6.0)))
        metrics = {
            'b_e': torch.tensor(2.0),
            'g_e': torch.tensor(0.2),
            'latency': torch.tensor(0.1),
            'cost': torch.tensor(0.1),
        }
        applied_rem = self.operator.remove_synapse('s2', metrics, self.paths_stats)
        self.assertTrue(applied_rem)
        self.assertNotIn('s2', self.graph.synapses)
        rem_pre = main.Reporter.report('complexity_before')
        rem_post = main.Reporter.report('complexity_after')
        self.assertTrue(torch.allclose(rem_pre, torch.tensor(6.0)))
        self.assertTrue(torch.allclose(rem_post, torch.tensor(5.0)))

    def test_move_synapse(self):
        metrics = {
            'b_e': torch.tensor(1.0),
            'g_e': torch.tensor(1.0),
            'latency': torch.tensor(0.1),
            'cost': torch.tensor(0.1),
        }
        applied = self.operator.move_synapse('s1', 'n2', 'n1', metrics, self.paths_stats)
        self.assertTrue(applied)
        src, tgt, _ = self.graph.synapses['s1']
        self.assertEqual(src, 'n2')
        self.assertEqual(tgt, 'n1')

    def test_add_neuron_threshold(self):
        ctx = {
            'neuron_id': 'n_fail',
            'neuron': Neuron(zero=self.zero),
            'source_id': 'n1',
            'target_id': 'n2',
            'synapse_in_id': 's_in_f',
            'synapse_in': Synapse(zero=self.zero),
            'synapse_out_id': 's_out_f',
            'synapse_out': Synapse(zero=self.zero),
            'b_vn': torch.tensor(0.1),
            'g_vn': torch.tensor(0.1),
            'latency': torch.tensor(0.5),
            'cost': torch.tensor(0.5),
        }
        applied = self.operator.add_neuron(ctx, self.paths_stats)
        self.assertFalse(applied)
        self.assertNotIn('n_fail', self.graph.neurons)


if __name__ == '__main__':
    unittest.main()
