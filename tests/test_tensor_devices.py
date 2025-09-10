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
