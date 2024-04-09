"""GPAW Response core functionality."""
from __future__ import annotations

from .groundstate import ResponseGroundStateAdapter, GPWFilename  # noqa
from .context import ResponseContext, TXTFilename, timer  # noqa

__all__ = ['ResponseGroundStateAdapter', 'GPWFilename',
           'ResponseContext', 'TXTFilename', 'timer']


class NoContext:
    def __init__(self):
        from gpaw.utilities.timing import NullTimer
        self.timer = NullTimer()

    def print(self, *args, **kwargs):
        pass


def read_ground_state(gpw: GPWFilename,
                      context: ResponseContext | NoContext | None = None
                      ) -> ResponseGroundStateAdapter:
    """Read ground state .gpw file.

    Logs and times the process if an actual ResponseContext is supplied."""
    if context is None:
        context = NoContext()
    context.print('Reading ground state calculation:\n  %s' % gpw)
    with context.timer('Read ground state'):
        return ResponseGroundStateAdapter.from_gpw_file(gpw)


def ensure_gs_and_context(gs: ResponseGroundStateAdapter | GPWFilename,
                          context: ResponseContext | TXTFilename = '-')\
        -> tuple[ResponseGroundStateAdapter, ResponseContext]:
    return ensure_gs(gs), ensure_context(context)


def ensure_gs(gs: ResponseGroundStateAdapter | GPWFilename
              ) -> ResponseGroundStateAdapter:
    if not isinstance(gs, ResponseGroundStateAdapter):
        gs = ResponseGroundStateAdapter.from_gpw_file(gpw=gs)
    return gs


def ensure_context(context: ResponseContext | TXTFilename
                   ) -> ResponseContext:
    if not isinstance(context, ResponseContext):
        context = ResponseContext(txt=context)
    return context
