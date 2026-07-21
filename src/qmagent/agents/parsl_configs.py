"""Parsl configurations for the QMAgent execution layer.

The QM workflow apps in ``distributed.py`` are labelled ``executors=['gpu']``
(the DFT-bound geometry/ESP/scan apps) and ``executors=['cpu']`` (build, RESP,
paramfit). Every config here must therefore define **both** a 'cpu' and a 'gpu'
executor, or parsl task routing raises at submit time.

Two configs, one axis of difference -- how many GPU tasks run at once:

* ``local_single_gpu_config`` -- the default. A ``ThreadPoolExecutor`` with a
  single 'gpu' slot: GPU tasks run one at a time in the agent's own process, on
  whatever single device ``CUDA_VISIBLE_DEVICES`` exposed to it. This is the
  right thing on a one-GPU allocation, and it is what ``QMAgent`` falls back to
  when no ``parsl_config`` is passed.

* ``multi_gpu_config`` -- fan GPU tasks out across N devices, one device per
  task, genuinely concurrently. This is what turns "4 torsion scans, ~105 min
  each, run back to back (~7 h)" into "4 scans at once (~105 min)".

Why the multi-GPU path needs a *process*-per-worker executor, not just more
threads:

    gpu4pyscf/CuPy bind to a device via ``CUDA_VISIBLE_DEVICES``, which CUDA
    reads once, per process, at context init. Threads in one process share one
    CUDA context and therefore one device -- so a ``ThreadPoolExecutor(max_threads=4)``
    would run four SCFs fighting over a *single* GPU, not spread across four.
    ``HighThroughputExecutor`` runs each worker in its own process, and
    ``available_accelerators`` makes parsl set that worker's own
    ``CUDA_VISIBLE_DEVICES`` before it starts -- distinct processes, distinct
    contexts, one GPU each. The app bodies need no change: each worker already
    sees exactly one device, so ``from gpu4pyscf import dft`` just works.
"""

from __future__ import annotations

import os

from parsl.config import Config
from parsl.executors import HighThroughputExecutor, ThreadPoolExecutor
from parsl.providers import LocalProvider


def local_single_gpu_config(num_threads: int) -> Config:
    """Single-slot local config: GPU tasks run one at a time, in-process.

    Defines 'cpu' and 'gpu' executors (both plain thread pools) to match the
    executor labels the apps in ``distributed.py`` request. The 'gpu' pool has a
    single slot, so GPU-labelled apps serialize; they still require gpu4pyscf +
    CUDA at runtime, so on a non-GPU host they fail at import inside the task.

    This is the historical default (previously ``_local_parsl_config`` in
    ``qm_agent.py``) and remains the right choice on a one-GPU allocation.

    Arguments:
        num_threads (int): Max threads for the 'cpu' executor.

    Returns:
        (Config): A parsl config with serial 'gpu' and threaded 'cpu' executors.
    """
    return Config(executors=[
        ThreadPoolExecutor(label='cpu', max_threads=num_threads),
        ThreadPoolExecutor(label='gpu', max_threads=1),
    ])


def multi_gpu_config(gpu_ids: list[str],
                     cpu_threads_per_worker: int | None = None,
                     total_cpu_threads: int | None = None) -> Config:
    """Fan GPU tasks across one device each, concurrently.

    Builds a ``HighThroughputExecutor`` for the 'gpu' label with one worker per
    entry in ``gpu_ids`` and ``available_accelerators`` set so parsl pins each
    worker process to a distinct device. N torsion scans (one parsl task each)
    then run on N GPUs at once instead of queueing on a single slot.

    IMPORTANT -- ``gpu_ids`` are **absolute physical device IDs**, not indices
    into ``CUDA_VISIBLE_DEVICES``. parsl's worker pool counts the node's GPUs
    with ``nvidia-smi -L`` (which ignores ``CUDA_VISIBLE_DEVICES``) and, when the
    worker count is <= the physical device count, sets each worker's
    ``CUDA_VISIBLE_DEVICES`` to the accelerator string *verbatim*
    (process_worker_pool.py). So ``['3','5','6','7']`` pins workers to physical
    GPUs 3,5,6,7 regardless of what the parent process had visible. The parent
    must be able to *see* those devices (do not launch under a
    ``CUDA_VISIBLE_DEVICES`` that hides them), but you do NOT renumber them --
    pass the physical IDs you actually want.

    Note for code-execution-only deployments: ``QMAgent.execute_code`` runs the
    snippet in a *subprocess*, which cannot reach the server-side agent's parsl.
    The fan-out config must therefore be built *inside the executed snippet*
    (which constructs its own ``QMAgent(parsl_config=multi_gpu_config(...))``),
    not on the server handle. Passing ``--gpus`` to the MCP server in that mode
    only spins up idle server-side workers that fight the snippet's for the same
    devices -- leave it off and let the snippet own the fan-out.

    Arguments:
        gpu_ids (list[str]): Absolute physical GPU IDs to dedicate one worker to
            each (e.g. ``['3','5','6','7']``). Must be non-empty. The launching
            process must be able to see all of them.
        cpu_threads_per_worker (int | None): ``lib.num_threads`` each GPU worker
            should use. Callers should thread this into the ``num_threads`` they
            pass to the QM apps so N workers don't oversubscribe the node. When
            None it is derived from ``total_cpu_threads`` / len(gpu_ids).
        total_cpu_threads (int | None): Total CPU threads to divide across the
            workers when ``cpu_threads_per_worker`` is not given. Defaults to
            ``os.cpu_count()``.

    Returns:
        (Config): A parsl config whose 'gpu' executor runs len(gpu_ids) workers,
            one pinned GPU each, plus a threaded 'cpu' executor for the
            CPU-bound apps (build/RESP/paramfit).

    Raises:
        ValueError: If ``gpu_ids`` is empty.
    """
    if not gpu_ids:
        raise ValueError('multi_gpu_config requires at least one GPU id.')

    n = len(gpu_ids)
    if cpu_threads_per_worker is None:
        total = total_cpu_threads or os.cpu_count() or n
        cpu_threads_per_worker = max(1, total // n)

    gpu_executor = HighThroughputExecutor(
        label='gpu',
        max_workers_per_node=n,
        available_accelerators=list(gpu_ids),
        # Keep each worker's OpenMP/MKL threads on their own cores; without this
        # N workers' PySCF thread pools collide and thrash a shared core set.
        cpu_affinity='block',
        provider=LocalProvider(init_blocks=1, min_blocks=1, max_blocks=1),
    )

    # A HighThroughputExecutor for the CPU-labelled apps too, so build/RESP/
    # paramfit don't block on the GPU pool and can themselves run concurrently.
    cpu_executor = HighThroughputExecutor(
        label='cpu',
        max_workers_per_node=n,
        cpu_affinity='block',
        provider=LocalProvider(init_blocks=1, min_blocks=1, max_blocks=1),
    )

    return Config(executors=[cpu_executor, gpu_executor])
