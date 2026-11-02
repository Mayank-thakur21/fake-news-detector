from backend.core.rate_limit import SlidingWindowLimiter


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def test_allows_up_to_limit_then_blocks():
    clock = Clock()
    lim = SlidingWindowLimiter(60, clock=clock)
    results = [lim.hit("k", 3) for _ in range(4)]
    assert [r[0] for r in results] == [True, True, True, False]
    assert [r[1] for r in results[:3]] == [2, 1, 0]


def test_window_slides():
    clock = Clock()
    lim = SlidingWindowLimiter(60, clock=clock)
    lim.hit("k", 1)
    assert lim.hit("k", 1)[0] is False
    clock.t += 61
    assert lim.hit("k", 1)[0] is True


def test_keys_are_independent():
    lim = SlidingWindowLimiter(60, clock=Clock())
    assert lim.hit("a", 1)[0]
    assert lim.hit("b", 1)[0]
    assert not lim.hit("a", 1)[0]


def test_reset_time_reported():
    clock = Clock()
    lim = SlidingWindowLimiter(60, clock=clock)
    lim.hit("k", 1)
    clock.t += 20
    allowed, remaining, reset = lim.hit("k", 1)
    assert not allowed and remaining == 0
    assert reset == 40
