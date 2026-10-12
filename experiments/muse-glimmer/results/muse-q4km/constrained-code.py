def first_unique(events):
    seen = set()
    result = []
    for e in events:
        if 'id' not in e:
            raise KeyError('id')
        id_val = e['id']
        if id_val not in seen:
            seen.add(id_val)
            result.append(e)
    return result

