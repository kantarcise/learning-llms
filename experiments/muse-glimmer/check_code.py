"""Check a reviewed model answer. Run untrusted code only in a sandbox."""

import argparse
import runpy


def check_chunks(function):
    cases = [([], 2, []), ([1, 2, 3, 4], 2, [[1, 2], [3, 4]])]
    cases += [([1, 2, 3], 2, [[1, 2], [3]]), ([1], 5, [[1]])]
    cases += [([1, 2], 1, [[1], [2]])]
    for items, size, expected in cases:
        original = items.copy()
        assert function(items, size) == expected
        assert items == original
        assert function(iter(items), size) == expected
    for invalid in [0, -1, True, False, 1.5, "2", None]:
        try:
            function(iter([1, 2]), invalid)
        except ValueError:
            continue
        raise AssertionError(f"Expected ValueError for {invalid!r}")


def check_unique(function):
    assert function(iter([])) == []
    first = {"id": 0, "value": "first"}
    second = {"id": None, "value": "second"}
    duplicate = {"id": 0, "value": "duplicate"}
    last = {"id": "x", "value": "last"}
    events = [first, second, duplicate, last, second]
    original = [event.copy() for event in events]
    result = function(iter(events))
    assert isinstance(result, list)
    assert len(result) == 3
    assert all(a is b for a, b in zip(result, [first, second, last], strict=True))
    assert events == original
    try:
        function(iter([{}]))
    except KeyError:
        return
    raise AssertionError("Expected KeyError for missing id")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("bug-fix", "constrained-code"))
    parser.add_argument("answer", help="Path to reviewed Python source")
    args = parser.parse_args()
    namespace = runpy.run_path(args.answer)
    if args.task == "bug-fix":
        check_chunks(namespace["chunks"])
    else:
        check_unique(namespace["first_unique"])
    print(f"PASS: {args.task}")


if __name__ == "__main__":
    main()
