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


class DeviceBudgetExceeded(Exception):
    """Raised when a memory allocation would exceed the device budget."""

    def __init__(self, device, requested, budget):
        message = f"{device} budget exceeded: requested {requested}, budget {budget}"
        super().__init__(message)


class DeviceCapacityExceeded(Exception):
    """Raised when a memory allocation would exceed the device capacity."""

    def __init__(self, device, requested, capacity):
        message = f"{device} capacity exceeded: requested {requested}, capacity {capacity}"
        super().__init__(message)


class EvictionPolicy:
    def __init__(self, device):
        self.device = device

    def evict(self, amount):
        raise NotImplementedError('EvictionPolicy.evict must be implemented')


class DropPolicy(EvictionPolicy):
    def evict(self, amount):
        freed = min(amount, self.device.used)
        self.device.used -= freed
        Reporter.report(f'{self.device.name}_used', f'Used capacity on {self.device.name}', self.device.used)
        count = Reporter.report('evictions') or 0
        Reporter.report('evictions', 'Number of evictions performed', count + 1)


class RemapPolicy(EvictionPolicy):
    def evict(self, amount):
        remapped = min(amount, self.device.used)
        self.device.used -= remapped
        self.device.reserved += remapped
        Reporter.report(f'{self.device.name}_used', f'Used capacity on {self.device.name}', self.device.used)
        Reporter.report(f'{self.device.name}_reserved', f'Reserved capacity on {self.device.name}', self.device.reserved)
        count = Reporter.report('remaps') or 0
        Reporter.report('remaps', 'Number of memory remaps', count + 1)


class MemoryDevice:
    def __init__(self, name, capacity, budget=None, eviction_policy=None):
        self.name = name
        self.capacity = capacity
        self.budget = capacity if budget is None else budget
        policy_cls = eviction_policy or DropPolicy
        self.eviction_policy = policy_cls(self)
        self.used = 0
        self.reserved = 0

    def allocate(self, amount, reserve=0):
        if amount < 0 or reserve < 0:
            raise ValueError('Amount and reserve must be non-negative')
        request = amount + reserve
        if self.used + request > self.capacity:
            self.eviction_policy.evict(request)
            raise DeviceCapacityExceeded(self.name, request, self.capacity)
        if self.used + self.reserved + request > self.budget:
            self.eviction_policy.evict(request)
            raise DeviceBudgetExceeded(self.name, request, self.budget)
        self.used += amount
        self.reserved += reserve
        Reporter.report(f'{self.name}_used', f'Used capacity on {self.name}', self.used)
        Reporter.report(f'{self.name}_reserved', f'Reserved capacity on {self.name}', self.reserved)

    def free(self, amount, release=0):
        if amount < 0 or release < 0 or amount > self.used or release > self.reserved:
            raise ValueError('Amount and release must be within used and reserved range')
        self.used -= amount
        self.reserved -= release
        Reporter.report(f'{self.name}_used', f'Used capacity on {self.name}', self.used)
        Reporter.report(f'{self.name}_reserved', f'Reserved capacity on {self.name}', self.reserved)

    def available(self):
        return self.budget - self.used - self.reserved


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
            try:
                op.execute()
                time += op.duration
            except DeviceBudgetExceeded:
                count = Reporter.report('budget_exceeded') or 0
                Reporter.report('budget_exceeded', 'Number of operations exceeding budget', count + 1)
                raise
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
