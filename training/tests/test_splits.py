from fishnet.splits import SPLITS, partition, split_of


def test_split_is_deterministic():
    keys = [f"run_{i:05d}" for i in range(500)]
    assert [split_of(k) for k in keys] == [split_of(k) for k in keys]


def test_partition_is_disjoint_and_complete():
    keys = [f"run_{i:05d}" for i in range(2000)]
    parts = partition(keys, lambda k: k)
    assert sorted(sum(parts.values(), [])) == sorted(keys)
    assert set(parts) == set(SPLITS)


def test_partition_fractions_are_close_to_targets():
    keys = [f"run_{i:05d}" for i in range(20000)]
    parts = partition(keys, lambda k: k)
    assert abs(len(parts["test"]) / len(keys) - 0.15) < 0.01
    assert abs(len(parts["val"]) / len(keys) - 0.15) < 0.01


def test_existing_assignments_stable_under_growth():
    small = {k: split_of(k) for k in (f"a_{i}" for i in range(300))}
    grown = {k: split_of(k) for k in [*small, *(f"b_{i}" for i in range(300))]}
    assert all(grown[k] == v for k, v in small.items())
