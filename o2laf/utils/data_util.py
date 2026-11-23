import random
from functools import wraps


# retry random fetch data when failed to read
def retry_load_error(max_attempts=3):
    if max_attempts < 1:
        raise ValueError('max_attempts must be at least 1.')

    def decorator(func):
        @wraps(func)
        def wrapper(self, index):
            dataset_size = len(self)
            if dataset_size <= 0:
                raise IndexError('Cannot load from an empty dataset.')

            index = int(index)
            if not 0 <= index < dataset_size:
                raise IndexError(
                    f'Dataset index {index} is outside [0, {dataset_size}).'
                )

            last_err = None
            error_summaries = []
            candidate_index = index
            attempted_indices = set()
            attempt_limit = min(max_attempts, dataset_size)
            for attempt in range(attempt_limit):
                attempted_indices.add(candidate_index)
                try:
                    return func(self, candidate_index)
                except Exception as e:
                    last_err = e
                    error_summaries.append(
                        f'{candidate_index}: {type(e).__name__}: {e}'
                    )
                    if attempt + 1 == attempt_limit:
                        break

                    # Draw lazily so a successful first attempt has exactly
                    # the same RNG effects as an undecorated __getitem__.
                    # Rejection sampling is cheap here: attempt_limit is tiny
                    # relative to the web-video corpora this wraps.
                    while True:
                        candidate_index = random.randrange(dataset_size)
                        if candidate_index not in attempted_indices:
                            break

            recent_errors = '; '.join(error_summaries[-3:])
            raise RuntimeError(
                f'Dataset failed after {len(attempted_indices)} distinct '
                f'attempts. Last errors: {recent_errors}'
            ) from last_err
        return wrapper
    return decorator
