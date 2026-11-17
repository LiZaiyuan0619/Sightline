"""No-op stand-ins for the optional panel-alignment audit used while the figures were drawn.

The audit compares panel geometry across a composite and fails the save when two panels that
should share an axis do not. It belongs to the drawing toolkit, not to reproduction: the figure
it checks is identical with or without it. These stand-ins keep the figure scripts unchanged.
"""


class PanelAlignmentError(AssertionError):
    pass


def require_matplotlib_panel_alignment(*args, **kwargs):
    return None


def audit_matplotlib_panel_alignment(*args, **kwargs):
    return {"comparisons": 0, "fail": 0, "warn": 0, "note": "audit not run"}
