"""A deliberately broken function for the disposable agent trial."""


def chunks(items, chunk_size):
    if type(chunk_size) is not int or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    result = []
    buffer = []
    for item in items:
        buffer.append(item)
        if len(buffer) == chunk_size:
            result.append(buffer)
            buffer = []
    if buffer:
        result.append(buffer)
    return result
