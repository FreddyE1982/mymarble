import unittest
import sys
import pathlib
import torch

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
import main
from poted.tensor import TensorBuilder


class TestTensorBuilderDeviceSwitch(unittest.TestCase):
    def setUp(self):
        main.Reporter._metrics = {}

    def test_device_switch(self):
        tokens = [[1, 2, 3], [4, 5]]
        builder = TensorBuilder(device='cpu', reporter=main.Reporter)
        tensor_cpu = builder.from_tokens(tokens)
        print('CPU tensor:', tensor_cpu, 'device:', tensor_cpu.device)
        self.assertEqual(str(tensor_cpu.device), 'cpu')
        if torch.cuda.is_available():
            tensor_gpu = builder.from_tokens(tokens, device='cuda:0')
            print('GPU tensor:', tensor_gpu, 'device:', tensor_gpu.device)
            self.assertTrue(str(tensor_gpu.device).startswith('cuda'))
            tensor_back = builder.from_tokens(tokens)
            print('Tensor back on default device:', tensor_back.device)
            self.assertEqual(str(tensor_back.device), 'cpu')
        else:
            print('CUDA not available; GPU test skipped')
        print('Reported shape:', main.Reporter.report('tensor_shape'))
        self.assertEqual(main.Reporter.report('tensor_shape'), list(tensor_cpu.shape))
