from functools import wraps
from time import perf_counter
from typing import Callable


def timeit(func: Callable) -> Callable:
    """
    Decorator for timing function execution time.
    Args:
        func: The function to time.
    Returns:
        The wrapped function.
    """

    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = perf_counter()
        result = func(*args, **kwargs)
        end_time = perf_counter()
        total_time = end_time - start_time
        print(f"Function {func.__name__} took {total_time:.2f} seconds")
        return result

    return timeit_wrapper