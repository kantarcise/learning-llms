def chunks(items, chunk_size):
    if type(chunk_size) is not int or chunk_size <= 0:
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
