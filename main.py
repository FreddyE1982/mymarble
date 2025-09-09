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


class MemoryBlock:
    def __init__(self, start, size, free=True):
        self.start = start
        self.size = size
        self.free = free

    def __repr__(self):
        state = 'free' if self.free else 'used'
        return f'MemoryBlock(start={self.start}, size={self.size}, {state})'


class DropPolicy(EvictionPolicy):
    def evict(self, amount):
        target = min(amount, self.device.used)
        if target:
            self.device._free_amount(target)
        count = Reporter.report('evictions') or 0
        Reporter.report('evictions', 'Number of evictions performed', count + 1)


class RemapPolicy(EvictionPolicy):
    def evict(self, amount):
        target = min(amount, self.device.used)
        if target:
            self.device._free_amount(target)
            self.device.reserved += target
            self.device._update_metrics()
        count = Reporter.report('remaps') or 0
        Reporter.report('remaps', 'Number of memory remaps', count + 1)


class MemoryDevice:
    def __init__(self, name, capacity, budget=None, eviction_policy=None):
        self.name = name
        self.capacity = capacity
        self.budget = capacity if budget is None else budget
        if self.budget > self.capacity:
            count = Reporter.report('invalid_device_configs') or 0
            Reporter.report(
                'invalid_device_configs',
                'Number of device configurations where budget exceeds capacity',
                count + 1,
            )
            raise ValueError(
                f"Budget {self.budget} exceeds capacity {self.capacity} for {self.name}"
            )
        policy_cls = eviction_policy or DropPolicy
        self.eviction_policy = policy_cls(self)
        self.blocks = [MemoryBlock(0, capacity, True)]
        self.allocations = []
        self.reserved = 0
        self._update_metrics()

    @property
    def used(self):
        return sum(block.size for block in self.blocks if not block.free)

    def _update_metrics(self):
        Reporter.report(
            f'{self.name}_used',
            f'Used capacity on {self.name}',
            self.used,
        )
        Reporter.report(
            f'{self.name}_reserved',
            f'Reserved capacity on {self.name}',
            self.reserved,
        )
        free_blocks = [b.size for b in self.blocks if b.free]
        total_free = sum(free_blocks)
        largest_free = max(free_blocks) if free_blocks else 0
        Reporter.report(
            f'{self.name}_largest_free_block',
            f'Largest free block on {self.name}',
            largest_free,
        )
        fragmentation = 0 if total_free == 0 else 1 - (largest_free / total_free)
        Reporter.report(
            f'{self.name}_fragmentation_ratio',
            f'Memory fragmentation ratio on {self.name}',
            fragmentation,
        )

    def _find_block(self, amount):
        for block in self.blocks:
            if block.free and block.size >= amount:
                return block
        return None

    def allocate(self, amount, reserve=0):
        if amount < 0 or reserve < 0:
            raise ValueError('Amount and reserve must be non-negative')
        request = amount + reserve
        if self.used + self.reserved + request > self.capacity:
            self.eviction_policy.evict(request)
            count = Reporter.report('allocation_failures') or 0
            Reporter.report('allocation_failures', 'Number of failed allocations', count + 1)
            raise DeviceCapacityExceeded(self.name, request, self.capacity)
        if self.used + self.reserved + request > self.budget:
            self.eviction_policy.evict(request)
            count = Reporter.report('allocation_failures') or 0
            Reporter.report('allocation_failures', 'Number of failed allocations', count + 1)
            raise DeviceBudgetExceeded(self.name, request, self.budget)
        block = self._find_block(amount)
        if block is None:
            count = Reporter.report('allocation_failures') or 0
            Reporter.report('allocation_failures', 'Number of failed allocations', count + 1)
            raise DeviceCapacityExceeded(self.name, amount, self.capacity)
        index = self.blocks.index(block)
        start = block.start
        if block.size == amount:
            block.free = False
            new_block = block
        else:
            new_block = MemoryBlock(start, amount, False)
            block.start += amount
            block.size -= amount
            self.blocks.insert(index, new_block)
        self.allocations.append(new_block)
        self.reserved += reserve
        self._update_metrics()
        return new_block

    def _free_amount(self, amount):
        if amount < 0 or amount > self.used:
            raise ValueError('Amount must be within used range')
        remaining = amount
        while remaining > 0:
            if not self.allocations:
                raise ValueError('No allocations to free')
            block = self.allocations.pop()
            if block.size > remaining:
                new_free = MemoryBlock(block.start + block.size - remaining, remaining, True)
                block.size -= remaining
                idx = self.blocks.index(block)
                self.blocks.insert(idx + 1, new_free)
                self.allocations.append(block)
                remaining = 0
            elif block.size == remaining:
                block.free = True
                remaining = 0
            else:
                block.free = True
                remaining -= block.size
        self._update_metrics()

    def free(self, block, release=0):
        if release < 0 or release > self.reserved:
            raise ValueError('Release must be within reserved range')
        if block not in self.allocations:
            raise ValueError('Block not allocated on this device')
        self.allocations.remove(block)
        block.free = True
        self.reserved -= release
        self._update_metrics()

    def compact(self):
        self.blocks.sort(key=lambda b: b.start)
        i = 0
        while i < len(self.blocks) - 1:
            current = self.blocks[i]
            nxt = self.blocks[i + 1]
            if current.free and nxt.free:
                current.size += nxt.size
                del self.blocks[i + 1]
            else:
                i += 1
        self._update_metrics()

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
        self._allocated_blocks = {}
        self._tensors = {}

    def allocate(self, manager):
        allocated = []
        try:
            if hasattr(manager, 'register'):
                for dt, amount in self.memory_requirements.items():
                    tensor = type('Tensor', (), {})()
                    tensor.device = dt
                    tensor.nbytes = amount
                    manager.register(tensor)
                    self._tensors[dt] = tensor
                    allocated.append(dt)
            else:
                devices = manager.devices if hasattr(manager, 'devices') else manager
                for dt, amount in self.memory_requirements.items():
                    block = devices[dt].allocate(amount)
                    self._allocated_blocks[dt] = block
                    allocated.append(dt)
        except (DeviceBudgetExceeded, DeviceCapacityExceeded):
            if hasattr(manager, 'register'):
                for dt in allocated:
                    manager.unregister(self._tensors[dt])
                self._tensors = {}
            else:
                devices = manager.devices if hasattr(manager, 'devices') else manager
                for dt in allocated:
                    devices[dt].free(self._allocated_blocks[dt])
                self._allocated_blocks = {}
            raise

    def free(self, manager):
        if hasattr(manager, 'unregister'):
            for tensor in self._tensors.values():
                manager.unregister(tensor)
            self._tensors = {}
        else:
            devices = manager.devices if hasattr(manager, 'devices') else manager
            for dt, block in self._allocated_blocks.items():
                devices[dt].free(block)
            self._allocated_blocks = {}


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
                op.allocate(self)
                time += op.duration
                op.free(self)
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


class TensorLoadBalancer(Scheduler):
    def __init__(self, devices):
        super().__init__(devices)
        self._registry = {}

    def register(self, tensor):
        tid = id(tensor)
        if tid in self._registry:
            raise ValueError('Tensor already registered')
        size = getattr(tensor, 'nbytes', None)
        if size is None:
            size = getattr(tensor, 'size', None)
        if size is None:
            raise AttributeError('Tensor size not specified')
        dev_key = getattr(tensor, 'device', None)
        if dev_key is None:
            dev_key = getattr(tensor, 'device_type', None)
        if dev_key is None:
            raise AttributeError('Tensor device not specified')
        device = self.devices[dev_key] if isinstance(dev_key, str) else dev_key
        block = device.allocate(size)
        self._registry[tid] = {'device': device, 'block': block, 'size': size}
        Reporter.report('registered_tensors', 'Number of tensors currently registered', len(self._registry))
        return block

    def unregister(self, tensor):
        tid = id(tensor)
        if tid not in self._registry:
            raise KeyError('Tensor not registered')
        meta = self._registry.pop(tid)
        meta['device'].free(meta['block'])
        Reporter.report('registered_tensors', 'Number of tensors currently registered', len(self._registry))

    def isRegistered(self, tensor):
        return id(tensor) in self._registry

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
                        op.allocate(self)
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
                    op.free(self)
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
