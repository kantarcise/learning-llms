"""A deliberately broken function for the disposable agent trial."""


def chunks(items, chunk_size):
    return [
        items[i : i + chunk_size] for i in range(0, len(items) - chunk_size, chunk_size)
    ]
