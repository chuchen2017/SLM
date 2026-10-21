"""Spatial measures and topological relations used as supervision."""

from __future__ import annotations

POLYGON_TYPES = {"Polygon", "MultiPolygon"}
LINE_TYPES = {"LineString", "MultiLineString"}

# Ordered so the first hit wins. Shapely reports equals geometries as both
# contains and within, so equality has to be tested before those two.
_RELATIONS = (
    "equals",
    "disjoint",
    "touches",
    "contains",
    "within",
    "crosses",
    "overlaps",
    "intersects",
)


def topological_relation(left, right) -> str:
    """Name the relation of ``left`` to ``right`` with Shapely predicates.

    The notebook converted a DE-9IM string with a function whose last ``covers``
    check was not exclusive, so it overwrote ``within`` and ``contains``. Named
    predicates are the relation the US topology file and the benchmarks use.
    """
    for name in _RELATIONS:
        if getattr(left, name)(right):
            return name
    return "disjoint"


def de9im_to_nl(matrix: str) -> str:
    """Map a 9-character DE-9IM matrix to the same vocabulary as ``topological_relation``."""
    if len(matrix) != 9:
        raise ValueError(f"DE-9IM matrix must be 9 characters, got {matrix!r}.")

    def is_true(cell: str) -> bool:
        return cell in {"T", "1", "2"}

    interior_interior, interior_boundary, interior_exterior, boundary_interior, _, _, exterior_interior, _, _ = matrix
    if matrix == "TFFFTFFFT":
        return "equals"
    if not any(is_true(cell) for cell in (interior_interior, interior_boundary, boundary_interior)):
        return "disjoint"
    if not is_true(interior_interior) and (is_true(interior_boundary) or is_true(boundary_interior)):
        return "touches"
    if is_true(interior_interior) and not is_true(exterior_interior):
        return "contains"
    if is_true(interior_interior) and not is_true(interior_exterior):
        return "within"
    if is_true(interior_interior) and is_true(interior_exterior) and is_true(exterior_interior):
        return "crosses"
    if is_true(interior_interior):
        return "overlaps"
    return "intersects"


def format_measure(value: float, decimals: int) -> str:
    """Format a length, area, or distance the way the benchmarks write labels.

    Zero decimals become an integer (``12``). Positive decimals follow
    ``str(round(value, decimals))``, which is ``1828.0`` rather than ``1828.00``.
    """
    value = float(value)
    if decimals <= 0:
        return str(int(round(value)))
    return str(round(value, decimals))
