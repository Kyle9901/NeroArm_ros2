"""Explicit object geometry shared by perception and manipulation."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ObjectGeometry:
    """Object pose semantics in the base frame; surface is the grasp reference."""

    surface_xyz: tuple[float, float, float]
    center_xyz: tuple[float, float, float] | None = None
    size_xyz: tuple[float, float, float] | None = None
    local_desk_z: float | None = None
    height: float | None = None
    height_source: str = "unknown"

    def to_dict(self) -> dict:
        def xyz(value):
            if value is None:
                return None
            return {"x": value[0], "y": value[1], "z": value[2]}

        return {
            "surface": xyz(self.surface_xyz),
            "center": xyz(self.center_xyz),
            "size": xyz(self.size_xyz),
            "local_desk_z": self.local_desk_z,
            "height": self.height,
            "height_source": self.height_source,
        }
