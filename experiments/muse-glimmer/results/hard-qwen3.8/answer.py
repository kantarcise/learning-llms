```python
import math
from collections import OrderedDict


class ExpiringLRU:
    def __init__(self, capacity):
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 0:
            raise ValueError("capacity must be an int >= 0")
        self._capacity = capacity
        self._store = OrderedDict()  # key -> (value