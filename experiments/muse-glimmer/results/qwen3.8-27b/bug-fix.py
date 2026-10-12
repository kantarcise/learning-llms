def chunks(items, chunk_size):
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int):
        raise ValueError
    if chunk_size <= 0:
        raise ValueError

    result = []
    current = []
    for item in items:
        current.append(item)
        if len(current) == chunk_size:
            result.append(current)
            current = []
    if current:
        result.append(current)
    return result
