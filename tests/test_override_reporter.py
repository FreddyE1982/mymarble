import unittest
import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))

from poted.poted import PoTED


class DummyReporter:
    def report(self, *args, **kwargs):
        pass


class DummyDecoder:
    def __init__(self, reporter):
        self._reporter = reporter


class DummyBuilder:
    def __init__(self, reporter):
        self._reporter = reporter


class TestOverrideReporter(unittest.TestCase):
    def setUp(self):
        self.original = DummyReporter()
        self.temporary = DummyReporter()
        self.decoder = DummyDecoder(self.original)
        self.builder = DummyBuilder(self.original)
        self.poted = PoTED(reporter=self.original)

    def test_restore_reporter(self):
        with self.poted._override(
            decoder=self.decoder, tensor_builder=self.builder, reporter=self.temporary
        ):
            self.assertIs(self.decoder._reporter, self.temporary)
            self.assertIs(self.builder._reporter, self.temporary)
        print('Decoder reporter after context:', self.decoder._reporter)
        print('Builder reporter after context:', self.builder._reporter)
        self.assertIs(self.decoder._reporter, self.original)
        self.assertIs(self.builder._reporter, self.original)


if __name__ == '__main__':
    unittest.main()
