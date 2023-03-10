"""
Module with utilities for distributed training.
"""
# Import standard library
import os
import sys
import logging
from datetime import datetime

# Import distributed torch
import torch
import torch.distributed as dist


def init_ddp(rank, world_size, backend="nccl"):
    """
    Initialize distributed training.

    Args:
        rank (int): Rank of the current process.
        world_size (int): Number of processes.
        backend (str): Distributed backend.

    Returns:
        torch.distributed.ProcessGroup: Process group for distributed training.
    """

    # Initialize the process group
    dist.init_process_group(
        backend,
        rank=rank,
        world_size=world_size,
        init_method="env://",
        timeout=datetime.timedelta(0, 1800),
    )

    # Set the device
    torch.cuda.set_device(rank)


def if_main_process(func):
    """
    Decorator to run a function only on the main process.

    Args:
        func (function): Function to run.

    Returns:
        function: Decorated function.
    """

    def wrapper(*args, **kwargs):
        if dist.get_rank() == 0:
            return func(*args, **kwargs)

    return wrapper
