def first_unique(events):
    seen = {}
    result = []
    for event in events:
        try:
            id = event['id']
        except KeyError:
            raise KeyError("Missing 'id' key")
        if id not in seen:
            seen[id] = event
            result.append(event)
    if len(seen) != len(result):
        raise KeyError("Duplicate ids found")
    return result
