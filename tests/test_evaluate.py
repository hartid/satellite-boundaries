import numpy as np

from evaluate import split_tiles


def test_split_tiles_is_spatial_and_deterministic():
    tiles = np.repeat(np.arange(10), 5).reshape(10, 5)
    tiles[0, 0] = -1  # пиксель без разметки

    train, test = split_tiles(tiles, test_size=0.2, seed=1)
    assert not (train & test).any()
    assert not train[0, 0] and not test[0, 0]
    # каждый тайл целиком в одной из частей
    for t in range(10):
        in_tile = tiles == t
        assert train[in_tile].all() or test[in_tile].all()
    assert len(np.unique(tiles[test])) == 2

    train2, test2 = split_tiles(tiles, test_size=0.2, seed=1)
    assert np.array_equal(test, test2)
