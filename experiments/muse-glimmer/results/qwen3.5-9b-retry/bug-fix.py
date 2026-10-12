def chunks(items, chunk_size):
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive int")

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
