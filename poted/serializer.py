class GenericSerializer:
    def __init__(self, reporter=None):
        self._reporter = reporter

    def serialize(self, obj):
        buffer = bytearray()
        max_depth = self._encode(obj, buffer, 1)
        if self._reporter:
            previous = self._reporter.report('max_recursion_depth') or 0
            if max_depth > previous:
                self._reporter.report(
                    'max_recursion_depth',
                    'Maximum recursion depth encountered during serialization',
                    max_depth,
                )
        return bytes(buffer)

    def deserialize(self, stream):
        data = memoryview(stream)
        obj, _ = self._decode(data, 0)
        return obj

    def _encode(self, obj, buffer, depth):
        if obj is None:
            buffer.append(ord('n'))
            return depth
        if obj is True:
            buffer.append(ord('t'))
            return depth
        if obj is False:
            buffer.append(ord('f'))
            return depth
        if isinstance(obj, int):
            buffer.append(ord('i'))
            sign = 1 if obj < 0 else 0
            buffer.append(sign)
            value = -obj if obj < 0 else obj
            buffer.extend(self._encode_varint(value))
            return depth
        if isinstance(obj, float):
            buffer.append(ord('F'))
            struct = __import__('struct')
            buffer.extend(struct.pack('>d', obj))
            return depth
        if isinstance(obj, (bytes, bytearray)):
            buffer.append(ord('b'))
            b = bytes(obj)
            buffer.extend(self._encode_varint(len(b)))
            buffer.extend(b)
            return depth
        if isinstance(obj, str):
            buffer.append(ord('s'))
            data = obj.encode('utf-8')
            buffer.extend(self._encode_varint(len(data)))
            buffer.extend(data)
            return depth
        if isinstance(obj, list):
            buffer.append(ord('l'))
            buffer.extend(self._encode_varint(len(obj)))
            max_depth = depth
            for item in obj:
                child = self._encode(item, buffer, depth + 1)
                if child > max_depth:
                    max_depth = child
            return max_depth
        if isinstance(obj, dict):
            buffer.append(ord('d'))
            buffer.extend(self._encode_varint(len(obj)))
            max_depth = depth
            for key, value in obj.items():
                child_key = self._encode(key, buffer, depth + 1)
                if child_key > max_depth:
                    max_depth = child_key
                child_val = self._encode(value, buffer, depth + 1)
                if child_val > max_depth:
                    max_depth = child_val
            return max_depth
        raise TypeError('Unsupported type: %r' % type(obj))

    def _decode(self, data, index):
        t = chr(data[index])
        index += 1
        if t == 'n':
            return None, index
        if t == 't':
            return True, index
        if t == 'f':
            return False, index
        if t == 'i':
            sign = data[index]
            index += 1
            value, index = self._decode_varint(data, index)
            return (-value if sign else value), index
        if t == 'F':
            struct = __import__('struct')
            value = struct.unpack('>d', data[index:index + 8])[0]
            index += 8
            return value, index
        if t == 'b':
            length, index = self._decode_varint(data, index)
            value = bytes(data[index:index + length])
            index += length
            return value, index
        if t == 's':
            length, index = self._decode_varint(data, index)
            value = data[index:index + length].tobytes().decode('utf-8')
            index += length
            return value, index
        if t == 'l':
            length, index = self._decode_varint(data, index)
            result = []
            for _ in range(length):
                item, index = self._decode(data, index)
                result.append(item)
            return result, index
        if t == 'd':
            length, index = self._decode_varint(data, index)
            result = {}
            for _ in range(length):
                key, index = self._decode(data, index)
                value, index = self._decode(data, index)
                result[key] = value
            return result, index
        raise ValueError('Unknown type tag %r at index %d' % (t, index - 1))

    @staticmethod
    def _encode_varint(value):
        buf = bytearray()
        while True:
            to_write = value & 0x7F
            value >>= 7
            if value:
                buf.append(to_write | 0x80)
            else:
                buf.append(to_write)
                break
        return bytes(buf)

    @staticmethod
    def _decode_varint(data, index):
        shift = 0
        result = 0
        while True:
            byte = data[index]
            index += 1
            result |= (byte & 0x7F) << shift
            if not (byte & 0x80):
                break
            shift += 7
        return result, index
