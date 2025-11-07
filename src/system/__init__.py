# Package marker for src.system
# Export commonly used helpers for convenience
from .system import (
    main_loop,
    SystemShell,
    parse_and_execute,
    handle_args,
)

__all__ = [
    "main_loop",
    "SystemShell",
    "parse_and_execute",
    "handle_args",
]
