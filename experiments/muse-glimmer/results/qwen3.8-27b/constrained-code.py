def first_unique(events):
    seen = set()
    result = []
    for event in events:
        event_id = event["id"]
        if event_id not in seen:
            seen.add(event_id)
            result.append(event)
    return result
