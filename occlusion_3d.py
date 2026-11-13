"""Editable, video-local recipes for projected 3D occlusion masks."""
from __future__ import annotations

import copy
import json
import math
import re
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

from pipeline import normalize_filename_affix
from mask_io import read_mask_png, write_mask_png


UNITS = {"m": 1.0, "cm": 0.01, "mm": 0.001}


def convert_length(value: float, source_unit: str, target_unit: str) -> float:
    """Preserve the physical length when changing the displayed metric unit."""
    if source_unit not in UNITS or target_unit not in UNITS or not math.isfinite(value):
        raise ValueError("Lengths must be finite and use m, cm, or mm.")
    return float(Decimal(str(value)) * Decimal(str(UNITS[source_unit])) / Decimal(str(UNITS[target_unit])))


def new_id() -> str:
    return uuid4().hex


def reference_pixel_length(points) -> float:
    """Validate the minimum span, allowing coordinate roundoff at exactly 1 px."""
    pixels = math.dist(*points)
    if not math.isfinite(pixels):
        raise ValueError("Reference length must be finite.")
    tolerance = 4 * max(math.ulp(float(value)) for point in points for value in point)
    if abs(pixels - 1) <= tolerance:
        pixels = 1.0
    if pixels < 1:
        raise ValueError("Reference endpoints must be at least 1 px apart.")
    return pixels


@dataclass
class LengthReference:
    name: str
    length: float = 35.0
    unit: str = "cm"
    points: list[tuple[float, float]] = field(default_factory=list)
    id: str = field(default_factory=new_id)

    @property
    def pixels_per_metre(self) -> float:
        if self.unit not in UNITS or not math.isfinite(self.length) or self.length <= 0:
            raise ValueError(f"{self.name}: enter a positive reference length.")
        try:
            points = np.asarray(self.points, dtype=np.float64)
        except (ValueError, TypeError):
            raise ValueError(f"{self.name}: measure the reference in this video.") from None
        if points.shape != (2, 2) or not np.all(np.isfinite(points)):
            raise ValueError(f"{self.name}: measure the reference in this video.")
        try:
            pixels = reference_pixel_length(points)
        except ValueError as exc:
            raise ValueError(f"{self.name}: {exc}") from None
        metres = self.length * UNITS[self.unit]
        if metres <= 0:
            raise ValueError(f"{self.name}: reference length is too small.")
        scale = pixels / metres
        if not math.isfinite(scale):
            raise ValueError(f"{self.name}: reference scale must be finite.")
        return scale


@dataclass
class ObjectFace:
    mask: np.ndarray
    margin: float = 2.0
    unit: str = "cm"
    reference_id: str | None = None


@dataclass
class OcclusionObject:
    name: str
    top: ObjectFace
    bottom: ObjectFace
    color: str = "#06b6d4"
    included: bool = True
    same_reference: bool = False
    id: str = field(default_factory=new_id)


@dataclass
class ObjectWorkspace:
    width: int = 0
    height: int = 0
    references: list[LengthReference] = field(default_factory=list)
    objects: list[OcclusionObject] = field(default_factory=list)
    video_key: str | None = None

    def reference_for(self, obj: OcclusionObject, face: str) -> LengthReference:
        source = obj.top if face == "top" or obj.same_reference else obj.bottom
        reference = next((r for r in self.references if r.id == source.reference_id), None)
        if reference is None:
            raise ValueError(f"{face.title()}: select a reference.")
        return reference

    def margin_pixels(self, obj: OcclusionObject, face: str) -> float:
        item = getattr(obj, face)
        if item.unit not in UNITS or not math.isfinite(item.margin):
            raise ValueError(f"{face.title()}: enter a finite margin.")
        return item.margin * UNITS[item.unit] * self.reference_for(obj, face).pixels_per_metre

    def adjusted_face(self, obj: OcclusionObject, face: str) -> np.ndarray:
        item = getattr(obj, face)
        if item.mask.shape != (self.height, self.width) or not np.any(item.mask):
            raise ValueError(f"{face.title()}: draw a non-empty face.")
        result = apply_margin(item.mask, self.margin_pixels(obj, face))
        if not np.any(result):
            raise ValueError(f"{face.title()}: negative margin removes the entire face.")
        return result

    def render(self, obj: OcclusionObject) -> np.ndarray:
        return self.render_with_outline(obj)[0]

    def render_with_outline(self, obj: OcclusionObject) -> tuple[np.ndarray, list]:
        return build_envelope(self.adjusted_face(obj, "top"), self.adjusted_face(obj, "bottom"))

    def errors(self, obj: OcclusionObject) -> list[str]:
        errors = []
        for face in ("top", "bottom"):
            try:
                self.adjusted_face(obj, face)
            except ValueError as exc:
                errors.append(str(exc))
        return errors

    def for_new_video(self, width: int, height: int) -> "ObjectWorkspace":
        """Carry recipes forward, never carry a measured scale into another video."""
        result = copy.deepcopy(self)
        result.width, result.height = width, height
        result.video_key = None
        for reference in result.references:
            reference.points = []
        for obj in result.objects:
            for face in (obj.top, obj.bottom):
                face.mask = cv2.resize(face.mask, (width, height), interpolation=cv2.INTER_NEAREST)
        return result


def apply_margin(mask: np.ndarray, pixels: float) -> np.ndarray:
    """Round once at rasterization; distance fields avoid giant dilation kernels."""
    if not math.isfinite(pixels):
        raise ValueError("Margin must be finite.")
    binary = (mask > 0).astype(np.uint8)
    magnitude = abs(pixels)
    # Metric conversion can place an exact half pixel a few ULPs below its value.
    # Correct only floating-point noise; genuinely sub-half-pixel margins stay zero.
    radius = int(math.floor(magnitude + 0.5 + 4 * math.ulp(magnitude)))
    if radius == 0 or not np.any(binary):
        return binary.copy()
    if pixels > 0:
        distance = cv2.distanceTransform(1 - binary, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        return (distance <= radius).astype(np.uint8)
    padded = np.pad(binary, 1)
    distance = cv2.distanceTransform(padded, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[1:-1, 1:-1]
    return (distance > radius).astype(np.uint8)


def connect_faces(top: np.ndarray, bottom: np.ndarray) -> np.ndarray:
    return build_envelope(top, bottom)[0]


def build_envelope(top: np.ndarray, bottom: np.ndarray) -> tuple[np.ndarray, list]:
    """Use the same hull vertices for straight viewer edges and saved raster pixels."""
    if top.shape != bottom.shape or top.ndim != 2:
        raise ValueError("Both faces must match the video dimensions.")
    contours = []
    for mask in (top, bottom):
        found, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not found:
            raise ValueError("Both faces must be non-empty.")
        contours.extend(found)
    hull = cv2.convexHull(np.concatenate(contours))
    result = np.zeros_like(top, dtype=np.uint8)
    cv2.fillConvexPoly(result, hull, 1)
    return result, hull.reshape(-1, 2).tolist()


def save_bundle(workspace: ObjectWorkspace, folder: Path) -> None:
    """Export compatible final PNGs plus lossless, editable face recipes."""
    folder.mkdir(parents=True, exist_ok=True)
    source_folder = folder / "_3d_sources"
    source_folder.mkdir(exist_ok=True)
    previous_masks = set()
    manifest_path = folder / "masks_manifest.json"
    if manifest_path.exists():
        try:
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
            if isinstance(previous, dict) and "objects_3d" in previous and isinstance(previous.get("masks"), dict):
                previous_masks = set(previous["masks"])
        except (ValueError, OSError):
            pass
    manifest = {
        "format": "happycold_occlusion_masks_v1", "metadata_version": 3,
        "width": workspace.width, "height": workspace.height,
        "references": [vars(r) for r in workspace.references], "objects_3d": [], "masks": {},
        "video_key": workspace.video_key,
    }
    used_names = set()
    for obj in workspace.objects:
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", obj.id):
            raise ValueError("Invalid object ID.")
        name = normalize_filename_affix(obj.name) or "object"
        if name.split(".")[0].upper() in {"CON", "PRN", "AUX", "NUL", *[f"COM{i}" for i in range(1, 10)], *[f"LPT{i}" for i in range(1, 10)]}:
            name = f"object_{name}"
        candidate, number = name, 2
        while candidate.casefold() in used_names:
            candidate, number = f"{name}_{number}", number + 1
        used_names.add(candidate.casefold())
        entry = {k: v for k, v in vars(obj).items() if k not in {"top", "bottom"}}
        for key in ("top", "bottom"):
            face = getattr(obj, key)
            relative = f"_3d_sources/{obj.id}_{key}.png"
            if not write_mask_png(folder / relative, (face.mask > 0).astype(np.uint8) * 255):
                raise ValueError(f"Could not write the {key} face of {obj.name}.")
            entry[key] = {"file": relative, "margin": face.margin, "unit": face.unit,
                          "reference_id": face.reference_id}
        manifest["objects_3d"].append(entry)
        if not workspace.errors(obj):
            if not write_mask_png(folder / f"{candidate}.png", workspace.render(obj) * 255):
                raise ValueError(f"Could not write the mask of {obj.name}.")
            manifest["masks"][candidate] = {
                "margin": 0, "margin_mode": "simple", "color": obj.color, "name": obj.name,
            }
    manifest_text = json.dumps(manifest, indent=2)
    manifest_path.write_text(manifest_text, encoding="utf-8")
    # The legacy 2D exporter shares this folder and replaces its top-level manifest.
    # Keep a separate editable recipe so exporting 2D masks cannot discard 3D sources.
    (source_folder / "objects_manifest.json").write_text(manifest_text, encoding="utf-8")
    # Remove only final PNGs owned by the previous 3D export, never unrelated masks.
    for stale in previous_masks - set(manifest["masks"]):
        filename = f"{stale}.png"
        target = folder / filename
        if target.name == filename and target.resolve().is_relative_to(folder.resolve()) and target.is_file():
            target.unlink()


def load_bundle(folder: Path, width: int, height: int, *, video_key: str | None = None) -> ObjectWorkspace:
    primary = folder / "masks_manifest.json"
    data = json.loads(primary.read_text(encoding="utf-8")) if primary.exists() else None
    if not isinstance(data, dict) or not isinstance(data.get("objects_3d"), list):
        backup = folder / "_3d_sources" / "objects_manifest.json"
        data = json.loads(backup.read_text(encoding="utf-8")) if backup.exists() else None
    if not isinstance(data, dict) or not isinstance(data.get("objects_3d"), list):
        raise ValueError("This folder has no editable 3D objects. Import a PNG into a face instead.")
    source_width, source_height = int(data["width"]), int(data["height"])
    if source_width <= 0 or source_height <= 0:
        raise ValueError("Invalid source video dimensions.")
    result = ObjectWorkspace(width, height, video_key=video_key)
    same_video = video_key is not None and data.get("video_key") == video_key
    for entry in data.get("references", []):
        reference = LengthReference(str(entry["name"]), float(entry["length"]), str(entry["unit"]),
                                    id=str(entry["id"]))
        if same_video and (width, height) == (source_width, source_height):
            reference.points = [tuple(map(float, p)) for p in entry.get("points", [])]
        result.references.append(reference)
    root = folder.resolve()
    for entry in data["objects_3d"]:
        faces = {}
        for key in ("top", "bottom"):
            source = entry[key]
            path = (folder / source["file"]).resolve()
            if not path.is_relative_to(root):
                raise ValueError("Face files must be inside the mask folder.")
            mask = read_mask_png(path)
            if mask is None:
                raise ValueError(f"Missing {key} face: {path.name}")
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
            faces[key] = ObjectFace((mask > 0).astype(np.uint8), float(source["margin"]),
                                    str(source["unit"]), source.get("reference_id"))
        # Materialize the effective bottom reference from older linked-face bundles.
        if entry.get("same_reference", True):
            faces["bottom"].reference_id = faces["top"].reference_id
        result.objects.append(OcclusionObject(
            str(entry["name"]), faces["top"], faces["bottom"], str(entry.get("color", "#06b6d4")),
            bool(entry.get("included", True)), False, str(entry["id"]),
        ))
    names = [obj.name for obj in result.objects]
    if (len(set(names)) != len(names) or any(not name.strip() for name in names)
            or len({r.id for r in result.references}) != len(result.references)
            or len({o.id for o in result.objects}) != len(result.objects)):
        raise ValueError("Object names and IDs must be non-empty and unique.")
    if any(not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", item.id) for item in [*result.references, *result.objects]):
        raise ValueError("Invalid object or reference ID.")
    if any(r.unit not in UNITS or not math.isfinite(r.length) or r.length <= 0 for r in result.references):
        raise ValueError("Reference lengths must be positive and use m, cm, or mm.")
    if any(f.unit not in UNITS or not math.isfinite(f.margin) for o in result.objects for f in (o.top, o.bottom)):
        raise ValueError("Face margins must be finite and use m, cm, or mm.")
    return result
