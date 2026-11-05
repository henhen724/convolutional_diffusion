"""Workaround for a Sherlock-specific ``transformers`` import failure.

This cluster's ``py-pytorch/2.4.1_py312`` module reports its installed
package version (via ``importlib.metadata.version("torch")``) as
``"2.4.0a0+gitunknown"`` -- an internal build tag, not the module name's
"2.4.1". Under PEP 440, ``2.4.0a0`` (alpha 0 of the 2.4.0 release) sorts
*before* the plain release ``2.4.0``, so ``transformers``'s own
``is_torch_available()`` (``transformers/utils/import_utils.py``) computes
``version.parse("2.4.0a0+gitunknown") >= version.parse("2.4.0")`` as
``False`` and disables all Torch-backed model classes, even though the
actual installed PyTorch (2.4.1) is more than new enough.

Call :func:`patch_torch_version_for_transformers` *before* the first
``import transformers`` (or anything that imports it transitively) in any
process that needs Torch-backed ``transformers`` classes (e.g.
``CLIPVisionModelWithProjection``) on this cluster. It only rewrites what
``importlib.metadata.version("torch")`` returns to callers -- ``torch``
itself and every other package are untouched.
"""

from __future__ import annotations

import importlib.metadata

_TORCH_VERSION_OVERRIDE = "2.4.1"
_patched = False


def patch_torch_version_for_transformers(override: str = _TORCH_VERSION_OVERRIDE) -> None:
    """Make ``importlib.metadata.version("torch")`` report a clean release
    string, so libraries that gate on it (e.g. ``transformers``) don't
    misread Sherlock's dev-build torch version tag as "too old"."""
    global _patched
    if _patched:
        return

    _original_version = importlib.metadata.version

    def _patched_version(name: str, *args, **kwargs) -> str:
        if name == "torch":
            return override
        return _original_version(name, *args, **kwargs)

    importlib.metadata.version = _patched_version
    _patched = True
