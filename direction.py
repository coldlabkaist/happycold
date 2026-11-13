"""Read-only object masks and per-row heading classification in video pixels."""
from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from batch import raise_if_cancelled
from mask_io import read_mask_png, write_mask_png
from pipeline import normalize_filename_affix
from occlusion_3d import UNITS, LengthReference, apply_margin, build_envelope, load_bundle
from shared import adjust_mask_by_mode, dataframe_points_to_pixels


@dataclass(frozen=True)
class DirectionConfig:
    method: str = "ears"
    nose: str = "nose"
    anchor: str = "neck"
    left_ear: str = "ear_l"
    right_ear: str = "ear_r"
    angle: float = 15.0
    coordinates: str = "auto"
    distance_node: str = "nose"

    def required_bodyparts(self):
        if self.method not in {"ears", "neck", "body"}:
            raise ValueError("Choose a direction method.")
        parts = [self.nose, self.left_ear, self.right_ear] if self.method == "ears" else [self.nose, self.anchor]
        if not all(parts) or len(set(parts)) != len(parts):
            raise ValueError("Choose distinct bodyparts for the direction points.")
        if not math.isfinite(self.angle) or not 0 <= self.angle <= 90:
            raise ValueError("Angle margin must be between 0 and 90 degrees.")
        if self.coordinates not in {"auto", "pixels", "normalized"}:
            raise ValueError("Choose Auto, Pixels, or Normalized coordinates.")
        return parts


@dataclass
class DirectionRanges:
    object_margin: float = 0.0
    near_margin: float = 20.0
    mode: str = "pixels"
    unit: str = "cm"
    reference_id: str | None = None
    top_reference_id: str | None = None
    bottom_reference_id: str | None = None
    quad_points: list = field(default_factory=list)
    calibration_video_key: str | None = None

    def validate(self):
        if self.mode not in {"pixels", "geometric", "physical"}:
            raise ValueError("Choose Pixels, Perspective, or Physical ranges.")
        if not all(math.isfinite(v) for v in (self.object_margin, self.near_margin)):
            raise ValueError("Margins must be finite.")
        if self.near_margin < self.object_margin:
            raise ValueError("Near Margin must be at least Object Margin.")
        if self.mode != "physical" and max(abs(self.object_margin), abs(self.near_margin)) > 300:
            raise ValueError("Pixel and perspective margins must be between -300 and 300 px.")
        if self.mode == "physical" and self.unit not in UNITS:
            raise ValueError("Physical ranges must use m, cm, or mm.")


@dataclass
class DirectionMask:
    name: str
    mask: np.ndarray
    color: str = "#06b6d4"
    video_key: str | None = None
    ranges: DirectionRanges = field(default_factory=DirectionRanges)
    top: np.ndarray | None = None
    bottom: np.ndarray | None = None
    references: list[LengthReference] = field(default_factory=list)

    def resized(self, width, height):
        result = copy.deepcopy(self)
        old_height, old_width = self.mask.shape
        for key in ("mask", "top", "bottom"):
            value = getattr(result, key)
            if value is not None:
                setattr(result, key, cv2.resize(value, (width, height), interpolation=cv2.INTER_NEAREST))
        result.ranges.quad_points = [(x * width / old_width, y * height / old_height) for x, y in self.ranges.quad_points]
        if result.top is not None:
            result.mask = build_envelope(result.top, result.bottom)[0]
        return result

    def regions(self):
        self.ranges.validate()
        if not np.any(self.mask):
            raise ValueError("Draw a non-empty base mask in the source tab.")
        spec = self.ranges
        def reference_scale(key):
            ref = next((r for r in self.references if r.id == key), None)
            if ref is None:
                raise ValueError("Choose a measured reference in Calibration.")
            return UNITS[spec.unit] * ref.pixels_per_metre
        def expand(value):
            if self.top is not None:
                if spec.mode == "geometric":
                    raise ValueError("Use Pixels or Physical ranges for 3D faces.")
                scales = (reference_scale(spec.top_reference_id), reference_scale(spec.bottom_reference_id)) if spec.mode == "physical" else (1, 1)
                top, bottom = apply_margin(self.top, value * scales[0]), apply_margin(self.bottom, value * scales[1])
                return build_envelope(top, bottom)[0]
            if spec.mode == "physical":
                return apply_margin(self.mask, value * reference_scale(spec.reference_id))
            return adjust_mask_by_mode(self.mask, int(round(value)),
                "geometric" if spec.mode == "geometric" else "simple", spec.quad_points)
        inner, outer = expand(spec.object_margin), expand(spec.near_margin)
        if not np.any(inner):
            raise ValueError("Object Margin removes the entire object.")
        if np.any((inner > 0) & (outer == 0)):
            raise ValueError("Object range must be contained in the near range.")
        return inner, outer


@dataclass(frozen=True)
class DirectionPose:
    origin: np.ndarray
    nose: np.ndarray
    valid: np.ndarray
    coordinates_valid: np.ndarray
    points: dict[str, np.ndarray]
    reasons: np.ndarray
    distance: np.ndarray
    distance_valid: np.ndarray
    distance_reasons: np.ndarray
    distance_name: str
    nose_name: str = "nose"

    def subset(self, selection):
        return DirectionPose(self.origin[selection], self.nose[selection], self.valid[selection],
                             self.coordinates_valid[selection],
                             {name: points[selection] for name, points in self.points.items()},
                             self.reasons[selection], self.distance[selection], self.distance_valid[selection],
                             self.distance_reasons[selection], self.distance_name, self.nose_name)


def _append_reason(reasons, selected, reason):
    rows = np.flatnonzero(selected)
    reasons[rows] = np.where(reasons[rows] == "", reason, reasons[rows] + ";" + reason)


def prepare_direction_pose(df, config: DirectionConfig, width, height):
    parts = config.required_bodyparts()
    if not config.distance_node:
        raise ValueError("Choose a Distance Node.")
    points = {}
    valid = np.ones(len(df), dtype=bool)
    reasons = np.full(len(df), "", dtype=object)
    point_valid, point_reasons = {}, {}
    for part in dict.fromkeys([*parts, config.distance_node]):
        x_col, y_col = f"{part}.x", f"{part}.y"
        if x_col not in df or y_col not in df:
            raise ValueError(f"Missing coordinates for {part}.")
        if config.coordinates == "auto":
            value = dataframe_points_to_pixels(df, x_col, y_col, width, height).astype(float)
        else:
            value = df[[x_col, y_col]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
            if config.coordinates == "normalized":
                value = value * (width, height)
        finite = np.isfinite(value).all(axis=1)
        in_frame = (value[:, 0] >= 0) & (value[:, 0] < width) & (value[:, 1] >= 0) & (value[:, 1] < height)
        missing = np.isnan(value).any(axis=1)
        reason = np.full(len(df), "", dtype=object)
        _append_reason(reason, missing, f"missing_{part}")
        _append_reason(reason, ~finite & ~missing, f"invalid_{part}")
        _append_reason(reason, finite & ~in_frame, f"{part}_outside_frame")
        point_valid[part], point_reasons[part] = finite & in_frame, reason
        if part in parts:
            valid &= point_valid[part]
            rows = reason != ""
            reasons[rows] = np.where(reasons[rows] == "", reason[rows], reasons[rows] + ";" + reason[rows])
        points[part] = value
    nose = points[config.nose]
    origin = (points[config.left_ear] + points[config.right_ear]) / 2 if config.method == "ears" else points[config.anchor]
    with np.errstate(invalid="ignore", over="ignore"):
        has_heading = np.linalg.norm(nose - origin, axis=1) > 1e-8
    _append_reason(reasons, valid & ~has_heading, "zero_length_heading")
    return DirectionPose(origin, nose, valid & has_heading, valid, points, reasons,
        points[config.distance_node], point_valid[config.distance_node], point_reasons[config.distance_node], config.distance_node, config.nose)


def heading_points(df, config: DirectionConfig, width, height):
    pose = prepare_direction_pose(df, config, width, height)
    return pose.origin, pose.nose, pose.valid


def direction_label(state, reason=""):
    """Readable UI labels; CSV reasons retain stable machine-readable codes."""
    title = {"NearFacing": "Near / Facing", "NearNotFacing": "Near / Not Facing"}.get(state, state)
    if not reason or state in {"NearFacing", "NearNotFacing", "Far"}:
        return title
    details = []
    for code in reason.split(";"):
        if code.endswith("_inside_object_zone"):
            name = code.removesuffix("_inside_object_zone")
            details.append(name if state == "Overlap" else f"{name} inside object")
        elif code.endswith(("_inside_near_zone", "_outside_near_zone")) or code.startswith("heading_"):
            continue
        elif code.endswith("_inside_mask"):
            details.append(code.removesuffix("_inside_mask"))
        elif code.startswith("missing_"):
            details.append(f"{code.removeprefix('missing_')} missing")
        elif code.startswith("invalid_"):
            details.append(f"{code.removeprefix('invalid_')} invalid")
        elif code.endswith("_outside_frame"):
            details.append(f"{code.removesuffix('_outside_frame')} out of frame")
        else:
            details.append({"zero_length_heading": "zero-length heading",
                            "no_frame_data": "no frame data", "not_selected": "not selected"}.get(code, code))
    return f"{title} ({', '.join(details)})" if details else title


def mask_rectangles(mask):
    """Exact union of occupied pixel cells, merging equal scanline runs.

    Unlike simplified contours, this retains holes and single-pixel objects.
    Pixel centers have integer coordinates; cell edges lie half a pixel away.
    """
    binary = np.asarray(mask) > 0
    if binary.ndim != 2 or not np.any(binary):
        raise ValueError("Object masks must be non-empty 2D images.")
    rectangles, active = [], {}
    for y, row in enumerate(binary):
        changes = np.diff(np.r_[False, row, False].astype(np.int8))
        runs = set(zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1) - 1))
        for key in active.keys() - runs:
            rectangles.append(active[key])
        active = {key: [key[0] - 0.5, active[key][1] if key in active else y - 0.5,
                        key[1] + 0.5, y + 0.5] for key in runs}
    rectangles.extend(active.values())
    return np.asarray(rectangles, dtype=float)


class DirectionTarget:
    def __init__(self, mask):
        self.rectangles = mask_rectangles(mask)
        r = self.rectangles
        self.corners = np.stack([r[:, [0, 1]], r[:, [0, 3]], r[:, [2, 1]], r[:, [2, 3]]], axis=1)

    def _ray_hits(self, origins, vectors):
        lo = self.rectangles[None, :, :2]
        hi = self.rectangles[None, :, 2:]
        p, d = origins[:, None, :], vectors[:, None, :]
        parallel = np.abs(d) < 1e-12
        safe = np.where(parallel, 1, d)
        t1, t2 = (lo - p) / safe, (hi - p) / safe
        enter = np.where(parallel, -np.inf, np.minimum(t1, t2)).max(axis=2)
        leave = np.where(parallel, np.inf, np.maximum(t1, t2)).min(axis=2)
        aligned = (~parallel | ((p >= lo - 1e-9) & (p <= hi + 1e-9))).all(axis=2)
        return (aligned & (leave >= np.maximum(enter, 0) - 1e-9) & (leave > 1e-9)).any(axis=1)

    def contains_points(self, points, should_cancel=None):
        result = np.zeros(len(points), dtype=bool)
        chunk = max(1, min(512, 200000 // len(self.rectangles)))
        r = self.rectangles
        for start in range(0, len(points), chunk):
            raise_if_cancelled(should_cancel)
            p = points[start:start + chunk]
            result[start:start + chunk] = ((p[:, None, 0] >= r[None, :, 0] - 1e-9)
                & (p[:, None, 0] <= r[None, :, 2] + 1e-9)
                & (p[:, None, 1] >= r[None, :, 1] - 1e-9)
                & (p[:, None, 1] <= r[None, :, 3] + 1e-9)).any(axis=1)
        return result

    def classify(self, origin, nose, valid, angle, should_cancel=None):
        """Geometry-only facing values; DirectionZones adds states and reasons."""
        if not math.isfinite(angle) or not 0 <= angle <= 90:
            raise ValueError("Angle margin must be between 0 and 90 degrees.")
        result = np.full(len(nose), np.nan)
        indices = np.flatnonzero(valid)
        chunk = max(1, min(512, 200000 // len(self.rectangles)))
        radians = math.radians(angle)
        cosine, sine = math.cos(radians), math.sin(radians)
        for start in range(0, len(indices), chunk):
            raise_if_cancelled(should_cancel)
            selected = indices[start:start + chunk]
            n = nose[selected]
            d = n - origin[selected]
            d /= np.linalg.norm(d, axis=1)[:, None]
            r = self.rectangles
            inside = ((n[:, None, 0] >= r[None, :, 0] - 1e-9)
                      & (n[:, None, 0] <= r[None, :, 2] + 1e-9)
                      & (n[:, None, 1] >= r[None, :, 1] - 1e-9)
                      & (n[:, None, 1] <= r[None, :, 3] + 1e-9)).any(axis=1)
            if angle == 0:
                # A zero-width cone is a forward ray, including boundary touches.
                hit = self._ray_hits(n, d)
            else:
                delta = self.corners[None] - n[:, None, None, :]
                forward = (delta * d[:, None, None, :]).sum(axis=3)
                cross = delta[..., 1] * d[:, None, None, 0] - delta[..., 0] * d[:, None, None, 1]
                corner = (forward >= -1e-9) & (np.abs(cross) * cosine <= forward * sine + 1e-9)
                hit = corner.any(axis=(1, 2))
                left = np.column_stack([d[:, 0] * cosine - d[:, 1] * sine, d[:, 0] * sine + d[:, 1] * cosine])
                right = np.column_stack([d[:, 0] * cosine + d[:, 1] * sine, -d[:, 0] * sine + d[:, 1] * cosine])
                hit |= self._ray_hits(n, left) | self._ray_hits(n, right)
            result[selected[~inside]] = hit[~inside].astype(float)
        return result


class DirectionZones:
    def __init__(self, mask: DirectionMask):
        self.object_mask, self.near_mask = mask.regions()
        self.object = DirectionTarget(self.object_mask)
        self.near = self.object if np.array_equal(self.object_mask, self.near_mask) else DirectionTarget(self.near_mask)

    def evaluate(self, pose: DirectionPose, angle, should_cancel=None):
        states = np.full(len(pose.nose), "Unknown", dtype=object)
        reasons = pose.distance_reasons.copy()
        valid = pose.distance_valid
        overlap = valid & self.object.contains_points(pose.distance, should_cancel)
        far = valid & ~self.near.contains_points(pose.distance, should_cancel)
        near = valid & ~overlap & ~far
        name = pose.distance_name
        states[overlap], reasons[overlap] = "Overlap", f"{name}_inside_object_zone"
        states[far], reasons[far] = "Far", f"{name}_outside_near_zone"
        reasons[near] = pose.reasons[near]
        values = self.object.classify(pose.origin, pose.nose, near & pose.valid, angle, should_cancel)
        facing, away = values == 1, values == 0
        states[facing], states[away] = "NearFacing", "NearNotFacing"
        reasons[facing] = f"{name}_inside_near_zone;heading_intersects_object_zone"
        reasons[away] = f"{name}_inside_near_zone;heading_misses_object_zone"
        # A different distance node can be Near while the nose is inside the
        # object. No angle can distinguish facing from away at that origin.
        ambiguous = near & pose.valid & np.isnan(values)
        reasons[ambiguous] = f"{pose.nose_name}_inside_object_zone"
        unknown_near = near & (states == "Unknown")
        reasons[unknown_near] = f"{name}_inside_near_zone;" + reasons[unknown_near]
        return states, reasons


def build_direction_dataframe(df, config, masks, width, height, should_cancel=None):
    if not masks or len({item.name for item in masks}) != len(masks):
        raise ValueError("Choose at least one object with a unique name.")
    pose = prepare_direction_pose(df, config, width, height)
    result = df.copy()
    for item in masks:
        raise_if_cancelled(should_cancel)
        if item.mask.shape != (height, width):
            raise ValueError(f"{item.name}: mask dimensions must match the video.")
        states, reasons = DirectionZones(item).evaluate(pose, config.angle, should_cancel)
        result[f"{item.name}.direction"] = states
        result[f"{item.name}.direction_reason"] = reasons
    return result


def direction_mask_from_object(obj, references, video_key):
    """Copy raw faces and calibration, never the source object's margins."""
    top, bottom = obj.top.mask.copy(), obj.bottom.mask.copy()
    base = build_envelope(top, bottom)[0]
    spec = DirectionRanges(top_reference_id=obj.top.reference_id,
        bottom_reference_id=obj.top.reference_id if obj.same_reference else obj.bottom.reference_id,
        calibration_video_key=video_key)
    return DirectionMask(obj.name, base, obj.color, video_key, spec, top, bottom, copy.deepcopy(references))


def _references_from_data(entries):
    result = [LengthReference(str(e["name"]), float(e["length"]), str(e["unit"]),
        [tuple(map(float, p)) for p in e.get("points", [])], str(e["id"])) for e in entries]
    if len({r.id for r in result}) != len(result) or any(r.unit not in UNITS or not math.isfinite(r.length) or r.length <= 0 for r in result):
        raise ValueError("Invalid direction calibration references.")
    return result


def load_direction_masks(path: Path, width, height, video_key=None):
    try:
        return _load_direction_masks(Path(path), width, height, video_key)
    except (KeyError, TypeError, IndexError, ZeroDivisionError, AttributeError, OverflowError, cv2.error) as exc:
        raise ValueError(f"Invalid direction mask metadata: {exc}") from exc


def _load_direction_masks(path: Path, width, height, video_key=None):
    folder = path if path.is_dir() else path.parent
    primary = folder / "masks_manifest.json"
    data = json.loads(primary.read_text(encoding="utf-8")) if primary.exists() else {}
    if not isinstance(data, dict):
        raise ValueError("Invalid mask manifest.")
    metadata = data.get("masks", data)
    if not isinstance(metadata, dict):
        raise ValueError("Invalid mask metadata.")
    if "objects_3d" in data:
        target_key = video_key or data.get("video_key") or "unknown-video"
        ws = load_bundle(folder, width, height, video_key=target_key)
        objects = ws.objects
        if not path.is_dir():
            name = metadata.get(path.stem, {}).get("name", path.stem)
            objects = [obj for obj in objects if obj.name == name]
        result = [direction_mask_from_object(obj, ws.references, target_key) for obj in objects if obj.included]
        if not result:
            raise ValueError("No raw 3D objects found. Import the full editable object folder.")
        return result
    files = sorted(folder.glob("*.png")) if path.is_dir() else [path]
    result, names = [], set()
    root = folder.resolve()
    backup = folder / "_3d_sources" / "objects_manifest.json"
    backup_ws = None
    if backup.exists() and not data.get("direction_base_masks"):
        backup_data = json.loads(backup.read_text(encoding="utf-8"))
        backup_ws = load_bundle(folder, int(backup_data["width"]), int(backup_data["height"]),
                                video_key=backup_data.get("video_key"))
    for file in files:
        image = read_mask_png(file)
        if image is None:
            raise ValueError(f"Could not read {file.name}.")
        entry = metadata.get(file.stem, {})
        if not isinstance(entry, dict):
            raise ValueError(f"Invalid metadata for {file.name}.")
        sidecar = file.with_suffix(".json")
        if not entry and sidecar.exists():
            entry = json.loads(sidecar.read_text(encoding="utf-8"))
            if not isinstance(entry, dict):
                raise ValueError(f"Invalid metadata for {file.name}.")
        name = str(entry.get("name") or file.stem)
        if not name.strip() or name in names:
            raise ValueError("Object names must be non-empty and unique.")
        names.add(name)
        recovered = None
        if backup_ws is not None:
            source = next((obj for obj in backup_ws.objects if obj.name == name), None)
            if source is not None and image.shape == (backup_ws.height, backup_ws.width):
                try:
                    if np.array_equal(image > 0, backup_ws.render(source) > 0):
                        recovered = direction_mask_from_object(source, backup_ws.references, backup_ws.video_key or "unknown-video")
                except ValueError:
                    pass
        if recovered is not None:
            if (height, width) != image.shape or (video_key is not None and recovered.video_key != video_key):
                for ref in recovered.references:
                    ref.points = []
                recovered.ranges.calibration_video_key = None
            recovered = recovered.resized(width, height)
            recovered.video_key = video_key or recovered.video_key
            result.append(recovered)
            continue
        recipe = entry.get("direction_recipe")
        if data.get("direction_settings") and not data.get("direction_base_masks") and file.stem in metadata:
            raise ValueError("This older Direction export contains baked margins. Reload the original Occlusion mask or editable 3D object folder.")
        binary = cv2.resize((image > 0).astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
        # Source margins, including geometric and physical values, are ignored.
        item = DirectionMask(name, binary, str(entry.get("color", "#06b6d4")))
        item.ranges.quad_points = copy.deepcopy(entry.get("geometric_points", data.get("geometric_points", [])))
        if recipe is not None:
            if not isinstance(recipe, dict):
                raise ValueError(f"Invalid direction recipe for {name}.")
            item._direction_recipe_loaded = True
            item.video_key = recipe.get("video_key")
            settings = recipe.get("ranges", {})
            item.ranges = DirectionRanges(**{k: v for k, v in settings.items() if k in DirectionRanges.__dataclass_fields__})
            item.ranges.validate()
            old_width, old_height = data.get("width", image.shape[1]), data.get("height", image.shape[0])
            item.ranges.quad_points = [(x * width / old_width, y * height / old_height) for x, y in item.ranges.quad_points]
            item.references = _references_from_data(recipe.get("references", []))
            if (video_key is not None and item.ranges.calibration_video_key != video_key) or (old_width, old_height) != (width, height):
                for ref in item.references:
                    ref.points = []
                item.ranges.calibration_video_key = None
            if recipe.get("kind") == "3d":
                for key in ("top", "bottom"):
                    face_path = (folder / recipe[key]).resolve()
                    if not face_path.is_relative_to(root):
                        raise ValueError("Face files must be inside the mask folder.")
                    face = read_mask_png(face_path)
                    if face is None or face.shape != (old_height, old_width):
                        raise ValueError(f"Missing or incompatible raw {key} face for {name}.")
                    face = cv2.resize((face > 0).astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
                    setattr(item, key, face)
                item.mask = build_envelope(item.top, item.bottom)[0]
            if video_key is not None and (item.video_key or item.top is not None):
                item.video_key = video_key
        elif item.ranges.quad_points:
            old_width, old_height = data.get("width", image.shape[1]), data.get("height", image.shape[0])
            item.ranges.quad_points = [(x * width / old_width, y * height / old_height) for x, y in item.ranges.quad_points]
        if not np.any(item.mask):
            raise ValueError(f"{name}: the base mask is empty.")
        result.append(item)
    if not result:
        raise ValueError("No PNG masks found.")
    return result


def export_direction_masks(masks, folder: Path, config: DirectionConfig):
    if not masks:
        raise ValueError("Choose at least one mask to export.")
    if len({m.video_key for m in masks if m.video_key}) > 1:
        raise ValueError("Video-specific masks must belong to the same video.")
    if any(m.mask.shape != masks[0].mask.shape for m in masks):
        raise ValueError("Exported masks must have matching dimensions.")
    folder.mkdir(parents=True, exist_ok=True)
    height, width = masks[0].mask.shape
    scoped = next((m.video_key for m in masks if m.video_key), None)
    manifest = {"format": "happycold_occlusion_masks_v1", "metadata_version": 4,
        "width": width, "height": height, "masks": {}, "video_scoped": bool(scoped),
        "video_key": scoped, "direction_settings": vars(config), "direction_base_masks": True,
        "direction_result_format": "proximity_direction_v3"}
    used = set()
    for item in masks:
        item.ranges.validate()
        stem = normalize_filename_affix(item.name) or "object"
        if stem.split(".")[0].upper() in {"CON", "PRN", "AUX", "NUL", *[f"COM{i}" for i in range(1, 10)], *[f"LPT{i}" for i in range(1, 10)]}:
            stem = f"object_{stem}"
        candidate, index = stem, 2
        while candidate.casefold() in used:
            candidate, index = f"{stem}_{index}", index + 1
        used.add(candidate.casefold())
        if not write_mask_png(folder / f"{candidate}.png", (item.mask > 0).astype(np.uint8) * 255):
            raise ValueError(f"Could not write {item.name}.")
        recipe = {"kind": "3d" if item.top is not None else "2d", "video_key": item.video_key,
                  "ranges": {key: getattr(item.ranges, key) for key in DirectionRanges.__dataclass_fields__},
                  "references": [vars(ref) for ref in item.references]}
        if item.top is not None:
            source_folder = folder / "_direction_sources"
            source_folder.mkdir(exist_ok=True)
            for key in ("top", "bottom"):
                relative = f"_direction_sources/{candidate}_{key}.png"
                if not write_mask_png(folder / relative, (getattr(item, key) > 0).astype(np.uint8) * 255):
                    raise ValueError(f"Could not write {key} face of {item.name}.")
                recipe[key] = relative
        manifest["masks"][candidate] = {"name": item.name, "color": item.color, "margin": 0,
                                       "margin_mode": "simple", "direction_recipe": recipe}
    (folder / "masks_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
