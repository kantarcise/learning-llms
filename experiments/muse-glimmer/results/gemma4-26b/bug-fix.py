def chunks(items, chunk_size):
    if type(chunk_size) is not int or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive int")
    
    result = []
    current_chunk = []
    for item in items:
        current_chunk.append(item)
        if len(current_chunk) == chunk_size:
            result.append(current_chunk)
            current_chunk = []
    
    if current_chunk:
        result.append(current_chunk)
        
    return result
