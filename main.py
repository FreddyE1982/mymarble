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
    def __init__(self, name, memory_requirement, duration, device_types):
        self.name = name
        self.duration = duration
        if isinstance(device_types, str):
            device_types = [device_types]
        self.device_types = device_types
        if isinstance(memory_requirement, int):
            self.memory_requirements = {
                dt: memory_requirement for dt in device_types
            }
        elif isinstance(memory_requirement, dict):
            missing = set(device_types) - set(memory_requirement)
            if missing:
                raise ValueError(f'Missing memory requirements for devices: {missing}')
            self.memory_requirements = memory_requirement
        else:
            raise TypeError('memory_requirement must be int or dict')

    def allocate(self, devices):
        allocated = []
        try:
            for dt, amount in self.memory_requirements.items():
                devices[dt].allocate(amount)
                allocated.append(dt)
        except (DeviceBudgetExceeded, DeviceCapacityExceeded):
            for dt in allocated:
                devices[dt].free(self.memory_requirements[dt])
            raise

    def free(self, devices):
        for dt, amount in self.memory_requirements.items():
            devices[dt].free(amount)


class Scheduler:
    def __init__(self, devices):
        if isinstance(devices, MemoryDevice):
            devices = [devices]
        self.devices = {d.name: d for d in devices}
        self.queues = {name: [] for name in self.devices}

    def run(self, operations):
        time = 0
        busy = {name: 0 for name in self.devices}
        timelines = {name: [] for name in self.devices}
        for op in operations:
            try:
                start = time
                op.allocate(self.devices)
                time += op.duration
                op.free(self.devices)
                for dt in op.device_types:
                    timelines[dt].append((start, time))
                    busy[dt] += op.duration
            except DeviceBudgetExceeded:
                count = Reporter.report('budget_exceeded') or 0
                Reporter.report('budget_exceeded', 'Number of operations exceeding budget', count + 1)
                raise
        Reporter.report('makespan', 'Total execution time', time)
        for name in self.devices:
            idle = time - busy[name]
            Reporter.report(f'{name}_idle_time', f'Idle time on {name}', idle)
            Reporter.report(f'{name}_timeline', f'Execution timeline on {name}', timelines[name])
        return time

    def run_parallel(self, operations):
        self.queues = {name: [] for name in self.devices}
        for op in operations:
            for dt in op.device_types:
                self.queues[dt].append(op)
        time = 0
        ready = operations[:]
        in_progress = []
        timelines = {name: [] for name in self.devices}
        busy = {name: 0 for name in self.devices}
        while ready or in_progress:
            started = []
            for op in list(ready):
                if all(self.devices[dt].available() >= op.memory_requirements[dt] for dt in op.device_types):
                    try:
                        op.allocate(self.devices)
                    except DeviceBudgetExceeded:
                        count = Reporter.report('budget_exceeded') or 0
                        Reporter.report('budget_exceeded', 'Number of operations exceeding budget', count + 1)
                        raise
                    op.remaining = op.duration
                    op.start_time = time
                    in_progress.append(op)
                    ready.remove(op)
            time += 1
            completed = []
            for op in in_progress:
                op.remaining -= 1
                if op.remaining == 0:
                    op.free(self.devices)
                    for dt in op.device_types:
                        timelines[dt].append((op.start_time, time))
                        busy[dt] += op.duration
                        self.queues[dt].remove(op)
                    completed.append(op)
            for op in completed:
                in_progress.remove(op)
        Reporter.report('makespan', 'Total execution time', time)
        for name in self.devices:
            idle = time - busy[name]
            Reporter.report(f'{name}_idle_time', f'Idle time on {name}', idle)
            Reporter.report(f'{name}_timeline', f'Execution timeline on {name}', timelines[name])
        return time


class Application:
    def run(self):
        gpu = MemoryDevice('GPU', 1024)
        cpu = MemoryDevice('CPU', 2048)
        ops = [
            Operation('op1', 256, 3, 'GPU'),
            Operation('op2', 256, 3, 'CPU'),
        ]
        scheduler = Scheduler([gpu, cpu])
        scheduler.run_parallel(ops)
        print('Metrics:', Reporter.report(['GPU_used', 'CPU_used', 'makespan']))


if __name__ == '__main__':
    Application().run()
