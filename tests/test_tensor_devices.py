import unittest
import sys
import pathlib
import torch
import json
from poted.pipeline import JsonSerializer

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
import main


class DummyTensor:
    def __init__(self, size, device):
        self.nbytes = size
        self.device = device


class DummyGradFn:
    def __init__(self, variable, parents=None):
        self.variable = variable
        self.next_functions = [(p.grad_fn, 0) for p in (parents or [])]


class GraphTensor(DummyTensor):
    def __init__(self, size, device, parents=None):
        super().__init__(size, device)
        self.grad_fn = DummyGradFn(self, parents)


class TestTensorLoadBalancerDevice(unittest.TestCase):
    def setUp(self):
        main.Reporter._metrics = {}

    def test_preserves_full_device_identifier(self):
        devices = [main.MemoryDevice('cuda:0', 100), main.MemoryDevice('cuda:1', 100)]
        balancer = main.TensorLoadBalancer(devices)
        tensor = DummyTensor(10, torch.device('cuda:1'))
        balancer.register(tensor)
        self.assertTrue(balancer.isRegistered(tensor))
        registered_device = balancer._registry[id(tensor)]['device']
        print('Registered tensors:', main.Reporter.report('registered_tensors'))
        self.assertEqual(registered_device.name, 'cuda:1')
        balancer.unregister(tensor)


class TestJsonSerializerDeviceFallback(unittest.TestCase):
    def setUp(self):
        main.Reporter._metrics = {}

    def test_cuda_payload_cpu_env(self):
        if torch.cuda.is_available():
            self.skipTest('CUDA available, fallback not triggered')
        serializer = JsonSerializer()
        payload = {
            "__torch_tensor__": [1.0, 2.0],
            "dtype": str(torch.float32),
            "device": "cuda:0",
            "requires_grad": False,
        }
        stream = json.dumps(payload).encode('utf-8')
        tensor = serializer.deserialize(stream)
        print('Deserialized tensor device:', tensor.device)
        self.assertEqual(tensor.device.type, 'cpu')


class TestTensorDeviceSync(unittest.TestCase):
    def setUp(self):
        main.Reporter._metrics = {}

    def test_moves_related_tensors_to_same_device(self):
        devices = [main.MemoryDevice('cpu', 100), main.MemoryDevice('gpu', 100)]
        balancer = main.TensorLoadBalancer(devices)
        a = GraphTensor(10, 'cpu')
        b = GraphTensor(10, 'gpu')
        c = GraphTensor(20, 'gpu', parents=[a, b])
        balancer.register(a)
        balancer.register(b)
        before_a = balancer._registry[id(a)]['device'].name
        before_b = balancer._registry[id(b)]['device'].name
        print('Before move:', before_a, before_b)
        balancer.register(c)
        after_a = balancer._registry[id(a)]['device'].name
        after_b = balancer._registry[id(b)]['device'].name
        print('After move:', after_a, after_b)
        self.assertEqual(after_a, 'gpu')
        self.assertEqual(after_b, 'gpu')
