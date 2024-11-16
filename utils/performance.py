import time
import psutil


def compute_cpu_usage(func, *args, **kwargs):
    """
    Measures the CPU usage and execution time of a given function.

    Parameters:
        func (function): The function to execute.
        *args: Arguments to pass to the function.
        **kwargs: Keyword arguments to pass to the function.

    Returns:
        tuple: (result, running_time, cpu_usage)
            - result: The output of the executed function.
            - running_time: Execution time in seconds.
            - cpu_usage: CPU usage as a percentage.
    """
    process = psutil.Process()  # Current process
    cpu_before = process.cpu_percent(interval=None)  # Initial CPU usage
    start_time = time.time()

    # Run the function
    result = func(*args, **kwargs)

    end_time = time.time()
    cpu_after = process.cpu_percent(interval=None)  # Final CPU usage

    running_time = end_time - start_time
    cpu_usage = (cpu_after - cpu_before)  # Approximate CPU usage

    return result, running_time, cpu_usage
