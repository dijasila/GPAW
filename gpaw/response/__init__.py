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
        return ResponseGroundStateAdapter.new_from_gpw_file(gpw)


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
        gs = read_ground_state(gpw=gs, context=context)
    return gs
