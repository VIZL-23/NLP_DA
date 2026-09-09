"""TG-FEM package - custom modules registered into Ultralytics.

Usage (must be called before building any model whose YAML names TGFEM):

    from tgfem import register
    register()
"""

from .language import ContextTokenLearner, TextConditioner, TextEncoder
from .module import TGFEM

__all__ = ["TGFEM", "ContextTokenLearner", "TextConditioner", "TextEncoder", "register"]

_REGISTERED = False


def register() -> None:
    """Make TGFEM resolvable from a model YAML.

    Ultralytics' `parse_model` resolves a module name with `globals()[m]`,
    evaluated inside `ultralytics.nn.tasks`. Injecting the class into that
    module's namespace is therefore all that is required - no fork, no patch
    of the parser itself.

    Idempotent, so it is safe to call from every entry point.
    """
    global _REGISTERED
    if _REGISTERED:
        return

    import ultralytics.nn.tasks as tasks
    from ultralytics.nn.modules import CBAM

    tasks.TGFEM = TGFEM  # type: ignore[attr-defined]  # dynamic injection is the whole point - see docstring
    # CBAM ships with Ultralytics but is not exposed in the tasks namespace, so
    # a model YAML cannot name it either. The CBAM control config needs it.
    tasks.CBAM = CBAM  # type: ignore[attr-defined]
    _REGISTERED = True
