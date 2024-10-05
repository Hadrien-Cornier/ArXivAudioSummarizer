import time
from functools import wraps
from typing import Callable, Any


def retry_with_exponential_backoff(
    func: Callable[..., Any],
    max_retries: int = 5,
    initial_wait: float = 1,
    exponential_base: float = 2,
) -> Callable[..., Any]:
    @wraps(func)
    def wrapper(*args, **kwargs):
        wait_time = initial_wait
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                print(
                    f"Attempt {attempt + 1}/{max_retries} failed: {str(e)}. Retrying in {wait_time:.2f} seconds..."
                )
                time.sleep(wait_time)
                wait_time *= exponential_base

    return wrapper
