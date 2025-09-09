class Reporter:
    _metrics = {}

    @classmethod
    def report(cls, metricname, metricdescription=None, value=None):
        if isinstance(metricname, list):
            return [cls._metrics.get(name) for name in metricname]
        if value is not None:
            cls._metrics[metricname] = value
            return value
        return cls._metrics.get(metricname)


class MemoryDevice:
    def __init__(self, name, capacity):
        self.name = name
        self.capacity = capacity
        self.used = 0

    def allocate(self, amount):
        if amount < 0:
            raise ValueError('Amount must be non-negative')
        if self.used + amount > self.capacity:
            raise MemoryError(f'{self.name} capacity exceeded')
        self.used += amount
        Reporter.report(f'{self.name}_used', f'Used capacity on {self.name}', self.used)

    def free(self, amount):
        if amount < 0 or amount > self.used:
            raise ValueError('Amount must be within used range')
        self.used -= amount
        Reporter.report(f'{self.name}_used', f'Used capacity on {self.name}', self.used)

    def available(self):
        return self.capacity - self.used


class Operation:
    def __init__(self, name, memory_requirement, duration, device):
        self.name = name
        self.memory_requirement = memory_requirement
        self.duration = duration
        self.device = device

    def execute(self):
        self.device.allocate(self.memory_requirement)
        self.device.free(self.memory_requirement)


class Scheduler:
    def __init__(self, device):
        self.device = device

    def run(self, operations):
        time = 0
        for op in operations:
            op.execute()
            time += op.duration
        Reporter.report('makespan', 'Total execution time', time)
        return time


class Application:
    def run(self):
        gpu = MemoryDevice('VRAM', 1024)
        ops = [
            Operation('op1', 256, 1, gpu),
            Operation('op2', 512, 2, gpu),
        ]
        scheduler = Scheduler(gpu)
        scheduler.run(ops)
        print('Metrics:', Reporter.report(['VRAM_used', 'makespan']))


if __name__ == '__main__':
    Application().run()
