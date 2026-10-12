def chunks(items, chunk_size):
    if not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive int")
    result = []
    iterator = iter(items)
    while True:
        chunk = list(iterator)
        if not chunk:
            break
        result.append(chunk)
    return result
