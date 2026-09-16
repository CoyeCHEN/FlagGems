"""Registration table for the reducing and constant-writing ``_foreach_*`` ops.

These five operators do not fit the element-wise executor in
:mod:`flag_gems.utils.foreach`, whose contract is ``fn(x) -> y`` over matching
element positions:

* ``_foreach_max`` and ``_foreach_norm`` / ``_foreach_powsum`` *reduce* -- each
  input tensor collapses to a zero-dimensional output (measured: a ``[4]``
  input yields a ``[]`` result).
* ``_foreach_zero`` / ``_foreach_zero_`` write a constant and never read the
  input at all.

Rather than reshape the shared executor around them, each one delegates to the
Triton implementation the repository already has for the single-tensor operator
-- ``ops/max.py``, ``ops/vector_norm.py`` and ``ops/zero.py``.  That keeps the
core computation in Triton (no ATen fallback) while writing no new reduction
kernel, and it is the same "call the existing implementation" shape the
element-wise rows use when they borrow a scalar function.

The per-tensor loop here is a real cost: unlike the element-wise path, launch
count grows with list length.  It is accepted deliberately, because a fused
multi-tensor reduction would be a new kernel rather than a reuse of an existing
one, and correctness parity with ATen comes first.
"""

import logging
from typing import Any, List, Optional, Sequence

import torch

from flag_gems.ops.max import max as _gems_max
from flag_gems.ops.zero import _launch_zero_kernel
from flag_gems.utils.foreach import check_tensor_list

logger = logging.getLogger(__name__)


def _foreach_max(self: Sequence[torch.Tensor]) -> List[torch.Tensor]:
    """Per-tensor maximum; each result is zero-dimensional.

    ``ops/max.py::max`` is the repository's Triton reduction, reused as is.
    """
    logger.debug("GEMS _FOREACH_MAX")
    tensors = check_tensor_list(self)
    return [_gems_max(t) for t in tensors]


def _norm_one(t: torch.Tensor, ord_: Any, dtype: Optional[torch.dtype], root: bool):
    """``sum(|t| ** ord)`` with an optional final root, as a 0-dim tensor.

    ``_foreach_norm`` takes the root; ``_foreach_powsum`` stops before it
    (measured on ``[3, 4]`` with ``ord=2``: norm 5.0, powsum 25.0).  The
    arithmetic runs through FlagGems' own pointwise and reduction operators, so
    the computation stays on Triton kernels.
    """
    from flag_gems.ops.abs import abs as gems_abs
    from flag_gems.ops.pow import pow_tensor_scalar
    from flag_gems.ops.sum import sum as gems_sum

    work = t if dtype is None else t.to(dtype)
    ord_f = float(ord_)
    if work.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        work = work.to(torch.float32)
    mag = gems_abs(work)
    powed = mag if ord_f == 1.0 else pow_tensor_scalar(mag, ord_f)
    total = gems_sum(powed)
    if not root or ord_f == 1.0:
        return total
    return pow_tensor_scalar(total, 1.0 / ord_f)


def _foreach_norm(
    self: Sequence[torch.Tensor], ord=2, dtype=None
) -> List[torch.Tensor]:
    """Per-tensor vector norm of order ``ord``."""
    logger.debug("GEMS _FOREACH_NORM")
    tensors = check_tensor_list(self)
    return [_norm_one(t, ord, dtype, root=True) for t in tensors]


def _foreach_powsum(
    self: Sequence[torch.Tensor], ord=2, dtype=None
) -> List[torch.Tensor]:
    """Per-tensor ``sum(|x| ** ord)`` -- ``norm`` without the final root."""
    logger.debug("GEMS _FOREACH_POWSUM")
    tensors = check_tensor_list(self)
    return [_norm_one(t, ord, dtype, root=False) for t in tensors]


def _zero_one(t: torch.Tensor) -> None:
    """Zero one tensor in place, including non-contiguous views.

    ``ops/zero.py``'s kernel asserts contiguity, but ATen's ``_foreach_zero_``
    accepts any view (a strided slice must be zeroed in the original storage).
    A non-contiguous target is therefore zeroed through a dense staging buffer
    and copied back, mirroring what the element-wise executor does for gappy
    views.
    """
    if t.is_contiguous():
        _launch_zero_kernel(t)
        return
    staged = torch.empty_like(t, memory_format=torch.contiguous_format)
    _launch_zero_kernel(staged)
    t.copy_(staged)


def _foreach_zero_(self: Sequence[torch.Tensor]) -> None:
    """Zero every tensor in place, reusing ``ops/zero.py``'s Triton kernel.

    The schema returns ``()``; handing the list back would make the dispatcher
    reject the kernel.
    """
    logger.debug("GEMS _FOREACH_ZERO_")
    tensors = check_tensor_list(self)
    for t in tensors:
        _zero_one(t)
    return None


def _foreach_zero(self: Sequence[torch.Tensor]) -> List[torch.Tensor]:
    """Functional form: ``self_out`` aliases and is zeroed, matching ATen.

    The schema is ``(Tensor[] self) -> Tensor[] self_out``: PyTorch zeroes the
    inputs and returns them, so this is not a fresh allocation.
    """
    logger.debug("GEMS _FOREACH_ZERO")
    tensors = check_tensor_list(self)
    for t in tensors:
        _zero_one(t)
    return tensors


def registered_wrappers():
    """Map ATen key -> wrapper for ``_FULL_CONFIG`` and the dispatch tests."""
    return {
        "_foreach_max": _foreach_max,
        "_foreach_norm.Scalar": _foreach_norm,
        "_foreach_powsum.Scalar": _foreach_powsum,
        "_foreach_zero": _foreach_zero,
        "_foreach_zero_": _foreach_zero_,
    }
