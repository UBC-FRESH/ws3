"""
Targeted coverage tests for ws3.forest_helper — multiprocessing task batching
and function sanitization utilities.
"""

import functools

import pytest

from ws3.forest_helper import (
    auto_batch,
    choose_max_batch_factor,
    sanitize_func,
)


class TestChooseMaxBatchFactor:
    def test_single_worker(self):
        assert choose_max_batch_factor(1) == 2

    def test_two_workers(self):
        assert choose_max_batch_factor(2) == 2

    def test_four_workers(self):
        assert choose_max_batch_factor(4) == 4  # 2 < 4 <= 8

    def test_eight_workers(self):
        assert choose_max_batch_factor(8) == 4

    def test_sixteen_workers(self):
        assert choose_max_batch_factor(16) == 8

    def test_thirty_two_workers(self):
        assert choose_max_batch_factor(32) == 16

    def test_zero_workers(self):
        # 0 <= 2 branch returns 2
        assert choose_max_batch_factor(0) == 2


class TestAutoBatch:
    def test_empty_tasks(self):
        result = auto_batch([], 4)
        assert result == []

    def test_single_task(self):
        result = auto_batch(['task1'], 4)
        assert result == [['task1']]

    def test_few_tasks_than_workers(self):
        result = auto_batch(['a', 'b'], 8)
        assert len(result) == 2
        assert all(b in [['a'], ['b']] for b in result)

    def test_even_distribution(self):
        tasks = list(range(20))
        result = auto_batch(tasks, 4)
        # All tasks should be present
        flat = [t for batch in result for t in batch]
        assert sorted(flat) == tasks

    def test_with_size_fn(self):
        tasks = ['short', 'medium', 'long']
        size_fn = {'short': 1.0, 'medium': 5.0, 'long': 10.0}.__getitem__
        result = auto_batch(tasks, 2, size_fn=size_fn)
        flat = [t for batch in result for t in batch]
        assert sorted(flat) == sorted(tasks)

    def test_max_batch_factor(self):
        tasks = list(range(100))
        result = auto_batch(tasks, 4, max_batch_factor=8)
        flat = [t for batch in result for t in batch]
        assert sorted(flat) == tasks


class TestSanitizeFunc:
    def test_simple_function(self):
        def my_func(x):
            return x * 2
        sanitized = sanitize_func(my_func)
        assert sanitized(5) == 10

    def test_function_with_defaults(self):
        def my_func(x, y=3):
            return x + y
        sanitized = sanitize_func(my_func)
        assert sanitized(5) == 8
        assert sanitized(5, 2) == 7

    def test_partial_function(self):
        def my_func(a, b):
            return a * b
        partial = functools.partial(my_func, 3)
        sanitized = sanitize_func(partial)
        assert sanitized(4) == 12

    def test_partial_with_keywords(self):
        def my_func(a, b=2):
            return a + b
        partial = functools.partial(my_func, b=10)
        sanitized = sanitize_func(partial)
        assert sanitized(5) == 15

    def test_raises_for_non_function(self):
        with pytest.raises(TypeError, match="Don't know how to sanitize"):
            sanitize_func("not a function")

    def test_raises_for_non_function_type(self):
        class MyClass:
            pass
        with pytest.raises(TypeError, match="Don't know how to sanitize"):
            sanitize_func(MyClass())
