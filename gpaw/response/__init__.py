"""GPAW Response core functionality."""
from __future__ import annotations

from .groundstate import ResponseGroundStateAdapter, GPWFilename  # noqa
from .context import ResponseContext, TXTFilename, timer  # noqa

__all__ = ['ResponseGroundStateAdapter', 'GPWFilename',
           'ResponseContext', 'TXTFilename', 'timer']


def ensure_gs_and_context(gs: ResponseGroundStateAdapter | GPWFilename,
                          context: ResponseContext | TXTFilename = '-')\
        -> tuple[ResponseGroundStateAdapter, ResponseContext]:
    if not isinstance(context, ResponseContext):
        context = ResponseContext(txt=context)
    gs = ensure_gs(gs, context=context)
    return gs, context


def ensure_gs(gs: ResponseGroundStateAdapter | GPWFilename,
              context: ResponseContext | None = None)\
        -> ResponseGroundStateAdapter:
    if not isinstance(gs, ResponseGroundStateAdapter):
        if context is None:
            context = ResponseContext()
        gs = ResponseGroundStateAdapter.from_gpw_file(gs, context)
    return gs
