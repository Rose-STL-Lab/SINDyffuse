from __future__ import annotations
from typing import TYPE_CHECKING
__all__ = ['LearnedSINDyGuidance']

def __getattr__(name: str):
    if name == 'LearnedSINDyGuidance':
        from .guidance import LearnedSINDyGuidance
        return LearnedSINDyGuidance
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
if TYPE_CHECKING:
    from .guidance import LearnedSINDyGuidance