"""Convert a triangle-mesh PLY into USD or USDZ for NVIDIA Isaac Sim.

Run this with Isaac Sim's Python (which provides ``pxr``), for example:
    ./python.sh ply_to_usd.py input.ply -o mesh.usd \
        --gaussian-usdz gaussians.usdz \
        --scene-output scene.usda

The reader supports ASCII and binary little/big-endian PLY files containing
vertex positions and face index lists. Optional vertex normals and RGB colors
are transferred to the USD mesh. Polygonal faces are fan-triangulated.
"""
from __future__ import annotations

import argparse
import struct
import sys
import tempfile
from pathlib import Path

import numpy as np

PLY_TYPES = {
    "char": ("i1", "b"), "int8": ("i1", "b"),
    "uchar": ("u1", "B"), "uint8": ("u1", "B"),
    "short": ("i2", "h"), "int16": ("i2", "h"),
    "ushort": ("u2", "H"), "uint16": ("u2", "H"),
    "int": ("i4", "i"), "int32": ("i4", "i"),
    "uint": ("u4", "I"), "uint32": ("u4", "I"),
    "float": ("f4", "f"), "float32": ("f4", "f"),
    "double": ("f8", "d"), "float64": ("f8", "d"),
}


def _read_header(stream):
    first = stream.readline().decode("ascii", errors="strict").strip()
    if first != "ply":
        raise ValueError("Input is not a PLY file (missing 'ply' header)")
    fmt = None
    elements = []
    current = None
    while True:
        line = stream.readline()
        if not line:
            raise ValueError("Unexpected end of PLY header")
        text = line.decode("ascii", errors="strict").strip()
        fields = text.split()
        if not fields or fields[0] in ("comment", "obj_info"):
            continue
        if fields[0] == "format":
            fmt = fields[1]
        elif fields[0] == "element":
            current = {"name": fields[1], "count": int(fields[2]), "properties": []}
            elements.append(current)
        elif fields[0] == "property":
            if current is None:
                raise ValueError("PLY property appears before an element")
            if fields[1] == "list":
                current["properties"].append(("list", fields[2], fields[3], fields[4]))
            else:
                current["properties"].append(("scalar", fields[1], fields[2]))
        elif fields[0] == "end_header":
            break
    if fmt not in ("ascii", "binary_little_endian", "binary_big_endian"):
        raise ValueError(f"Unsupported PLY format {fmt!r}")
    by_name = {e["name"]: e for e in elements}
    if "vertex" not in by_name or "face" not in by_name:
        raise ValueError("PLY must contain vertex and face elements (a triangle mesh)")
    return fmt, elements


def _read_scalar_vertices_ascii(stream, element):
    props = element["properties"]
    if any(p[0] != "scalar" for p in props):
        raise ValueError("PLY vertex properties must be scalar")
    rows = np.empty((element["count"], len(props)), dtype=np.float64)
    for i in range(element["count"]):
        values = np.fromstring(stream.readline().decode("ascii"), sep=" ")
        if values.size != len(props):
            raise ValueError(f"Malformed PLY vertex record {i}")
        rows[i] = values
    return rows, [p[2] for p in props]


def _read_scalar_vertices_binary(stream, element, endian):
    props = element["properties"]
    if any(p[0] != "scalar" for p in props):
        raise ValueError("PLY vertex properties must be scalar")
    dtype = np.dtype([(name, endian + PLY_TYPES[typ][0]) for _, typ, name in props])
    records = np.fromfile(stream, dtype=dtype, count=element["count"])
    if len(records) != element["count"]:
        raise ValueError("Unexpected end of PLY vertex data")
    rows = np.column_stack([records[name] for _, _, name in props]).astype(np.float64, copy=False)
    return rows, [p[2] for p in props]


def _attribute(rows, names, aliases, required=False):
    indices = [names.index(name) if name in names else None for name in aliases]
    if any(i is None for i in indices):
        if required:
            raise ValueError(f"PLY vertex data is missing one of {aliases}")
        return None
    return rows[:, indices]


def _read_faces_ascii(stream, element):
    list_props = [p for p in element["properties"] if p[0] == "list"]
    if not list_props:
        raise ValueError("PLY face element has no vertex index list")
    prop = next((p for p in list_props if p[3] in ("vertex_indices", "vertex_index")), list_props[0])
    faces = []
    for i in range(element["count"]):
        tokens = stream.readline().decode("ascii").split()
        if not tokens:
            raise ValueError(f"Malformed PLY face record {i}")
        cursor = 0
        indices = None
        for p in element["properties"]:
            if p[0] == "scalar":
                cursor += 1
                continue
            n = int(tokens[cursor]); cursor += 1
            values = list(map(int, tokens[cursor:cursor + n])); cursor += n
            if p == prop:
                indices = values
        if indices is None or len(indices) < 3:
            raise ValueError(f"PLY face {i} has fewer than three vertices")
        faces.extend((indices[0], indices[j], indices[j + 1]) for j in range(1, len(indices) - 1))
    return np.asarray(faces, dtype=np.int32)


def _read_faces_binary(stream, element, endian):
    props = element["properties"]
    list_props = [p for p in props if p[0] == "list"]
    if not list_props:
        raise ValueError("PLY face element has no vertex index list")
    prop = next((p for p in list_props if p[3] in ("vertex_indices", "vertex_index")), list_props[0])
    # Fast path for the standard Open3D layout: one uchar count and three int indices.
    if len(props) == 1 and prop[1] in ("uchar", "uint8") and prop[2] in ("int", "int32"):
        dtype = np.dtype([("n", "u1"), ("v", endian + "i4", (3,))])
        records = np.fromfile(stream, dtype=dtype, count=element["count"])
        if len(records) != element["count"]:
            raise ValueError("Unexpected end of PLY face data")
        if np.all(records["n"] == 3):
            return records["v"].astype(np.int32, copy=False)
        # Reparse records if they are polygons (the fast path read only 3 indices).
        raise ValueError("Binary polygon faces must use a fixed triangle PLY layout")
    faces = []
    for i in range(element["count"]):
        indices = None
        for p in props:
            if p[0] == "scalar":
                code = PLY_TYPES[p[1]][1]
                stream.read(struct.calcsize(endian + code))
                continue
            _, count_type, item_type, name = p
            count_code, item_code = PLY_TYPES[count_type][1], PLY_TYPES[item_type][1]
            count_data = stream.read(struct.calcsize(endian + count_code))
            if len(count_data) != struct.calcsize(endian + count_code):
                raise ValueError("Unexpected end of PLY face data")
            n = struct.unpack(endian + count_code, count_data)[0]
            raw = stream.read(n * struct.calcsize(endian + item_code))
            if len(raw) != n * struct.calcsize(endian + item_code):
                raise ValueError("Unexpected end of PLY face data")
            values = list(struct.unpack(endian + item_code * n, raw))
            if p == prop:
                indices = values
        if indices is None or len(indices) < 3:
            raise ValueError(f"PLY face {i} has fewer than three vertices")
        faces.extend((indices[0], indices[j], indices[j + 1]) for j in range(1, len(indices) - 1))
    return np.asarray(faces, dtype=np.int32)


def load_ply_mesh(path: Path):
    with path.open("rb") as stream:
        fmt, elements = _read_header(stream)
        vertex = next(e for e in elements if e["name"] == "vertex")
        face = next(e for e in elements if e["name"] == "face")
        if elements.index(vertex) > elements.index(face):
            raise ValueError("PLY face element precedes vertex element; unsupported ordering")
        if fmt == "ascii":
            vertex_rows, names = _read_scalar_vertices_ascii(stream, vertex)
            faces = _read_faces_ascii(stream, face)
        else:
            endian = "<" if fmt == "binary_little_endian" else ">"
            vertex_rows, names = _read_scalar_vertices_binary(stream, vertex, endian)
            faces = _read_faces_binary(stream, face, endian)
    vertices = _attribute(vertex_rows, names, ("x", "y", "z"), required=True)
    normals = _attribute(vertex_rows, names, ("nx", "ny", "nz"))
    colors = _attribute(vertex_rows, names, ("red", "green", "blue"))
    if colors is not None and colors.size and colors.max() > 1.0:
        colors = colors / 255.0
    if not len(vertices) or not len(faces):
        raise ValueError("PLY contains no mesh geometry")
    if faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError("PLY face indices reference vertices outside the vertex array")
    if not np.isfinite(vertices).all():
        raise ValueError("PLY contains non-finite vertex positions")
    return vertices, faces, normals, colors


def write_usd(vertices, faces, normals, colors, output: Path, scale: float, up_axis: str, meters_per_unit: float):
    try:
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdUtils, Vt
    except ImportError as exc:
        raise SystemExit(
            "The Pixar USD Python bindings (pxr) are required. Run with Isaac Sim's "
            "python.sh, or install usd-core in the active environment."
        ) from exc

    output.parent.mkdir(parents=True, exist_ok=True)
    is_usdz = output.suffix.lower() == ".usdz"
    temp_dir = tempfile.TemporaryDirectory(prefix="ply_to_usd_") if is_usdz else None
    stage_path = Path(temp_dir.name) / "mesh.usda" if temp_dir else output
    stage = Usd.Stage.CreateNew(str(stage_path))
    if stage is None:
        raise RuntimeError(f"Could not create USD stage at {stage_path}")
    UsdGeom.SetStageUpAxis(stage, {"X": UsdGeom.Tokens.x, "Y": UsdGeom.Tokens.y, "Z": UsdGeom.Tokens.z}[up_axis])
    UsdGeom.SetStageMetersPerUnit(stage, meters_per_unit)
    world = UsdGeom.Xform.Define(stage, "/World")
    # USD references without an explicit prim path require a default prim.
    stage.SetDefaultPrim(world.GetPrim())
    mesh = UsdGeom.Mesh.Define(stage, "/World/Mesh")
    points_np = (vertices * scale).astype(np.float32)
    points = Vt.Vec3fArray.FromNumpy(points_np)
    mesh.CreatePointsAttr(points)
    lo, hi = points_np.min(axis=0), points_np.max(axis=0)
    mesh.CreateExtentAttr(
        Vt.Vec3fArray(
            [
                Gf.Vec3f(float(lo[0]), float(lo[1]), float(lo[2])),
                Gf.Vec3f(float(hi[0]), float(hi[1]), float(hi[2])),
            ]
        )
    )
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(faces), 3, dtype=np.int32)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces.reshape(-1).astype(np.int32)))
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateOrientationAttr(UsdGeom.Tokens.rightHanded)
    mesh.CreateDoubleSidedAttr(True)
    if normals is not None:
        normal_values = normals.astype(np.float32)
        if scale < 0:
            normal_values = -normal_values
        mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(normal_values))
        mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    if colors is not None:
        primvar = mesh.CreateDisplayColorPrimvar(UsdGeom.Tokens.vertex)
        primvar.Set(Vt.Vec3fArray.FromNumpy(np.clip(colors, 0, 1).astype(np.float32)))
    stage.GetRootLayer().Save()

    if is_usdz:
        if not UsdUtils.CreateNewUsdzPackage(str(stage_path), str(output)):
            raise RuntimeError(f"USDZ packaging failed for {output}")
        temp_dir.cleanup()
    return output


def write_combined_scene(
    gaussian_usdz: Path, mesh_usd: Path, output_usda: Path, gaussian_prim_path: str
):
    """Compose sibling assets and copy the selected Gaussian prim's world transform."""
    from pxr import Sdf, Usd, UsdGeom

    gaussian_stage = Usd.Stage.Open(str(gaussian_usdz))
    if gaussian_stage is None:
        raise RuntimeError(f"Could not open Gaussian USDZ: {gaussian_usdz}")
    output_usda.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(output_usda))
    if stage is None:
        raise RuntimeError(f"Could not create composed USDA: {output_usda}")
    up_axis = UsdGeom.GetStageUpAxis(gaussian_stage)
    UsdGeom.SetStageUpAxis(stage, up_axis)
    UsdGeom.SetStageMetersPerUnit(stage, UsdGeom.GetStageMetersPerUnit(gaussian_stage))

    UsdGeom.Xform.Define(stage, "/World")
    gaussians = UsdGeom.Xform.Define(stage, "/World/Gaussians").GetPrim()
    gaussians.GetReferences().AddReference(str(gaussian_usdz.resolve()))

    # Snapshot the selected prim's composed world transform, then author the
    # mesh as a sibling so its visibility is independent of the Gaussian scene.
    target_path = Sdf.Path(gaussian_prim_path)
    target = stage.GetPrimAtPath(target_path)
    if not target or not target.IsValid():
        raise RuntimeError(
            f"Gaussian scene does not contain {target_path}. Check the USDZ's composed prim paths."
        )
    # A sibling can be toggled/viewed without showing the Gaussian prim. Since
    # USD has no live parent-transform link for a detached prim, bake the
    # target's composed world matrix onto this mesh at the default time.
    world_matrix = UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(target)
    mesh_path = Sdf.Path("/World/Mesh")
    mesh = UsdGeom.Mesh.Define(stage, mesh_path).GetPrim()
    mesh.GetReferences().AddReference(str(mesh_usd.resolve()), Sdf.Path("/World/Mesh"))
    mesh_xform = UsdGeom.Xformable(mesh)
    mesh_xform.AddTransformOp(UsdGeom.XformOp.PrecisionDouble).Set(world_matrix)
    stage.GetRootLayer().Save()
    return output_usda


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Input triangle-mesh .ply file")
    parser.add_argument("-o", "--output", type=Path, help="Output .usd, .usda, .usdc, or .usdz file (default: same basename .usd)")
    parser.add_argument("--scale", type=float, default=1.0, help="Scale vertex positions; default unit is metres")
    parser.add_argument("--up-axis", choices=("X", "Y", "Z"), default="Z", help="USD stage up-axis metadata")
    parser.add_argument("--meters-per-unit", type=float, default=1.0, help="USD stage linear units in metres")
    parser.add_argument("--gaussian-usdz", type=Path, help="Optional Gaussian splatting scene; also writes a composed USDA")
    parser.add_argument("--scene-output", type=Path, help="Output composed USDA path (default: <mesh>_with_gaussians.usda)")
    parser.add_argument("--gaussian-prim", default="/World/Gaussians/gauss/gauss", help="Gaussian prim whose composed world transform is copied onto the independent mesh prim")
    args = parser.parse_args()
    source = args.input.expanduser().resolve()
    if not source.is_file():
        parser.error(f"input file does not exist: {source}")
    output = (args.output or source.with_suffix(".usd")).expanduser().resolve()
    if output == source:
        parser.error("output path must differ from the input PLY")
    if output.suffix.lower() not in (".usd", ".usda", ".usdc", ".usdz"):
        parser.error("output extension must be .usd, .usda, .usdc, or .usdz")
    if args.scale == 0 or args.meters_per_unit <= 0:
        parser.error("--scale must be nonzero and --meters-per-unit must be positive")
    if args.scene_output and not args.gaussian_usdz:
        parser.error("--scene-output requires --gaussian-usdz")
    gaussian_usdz = args.gaussian_usdz.expanduser().resolve() if args.gaussian_usdz else None
    if gaussian_usdz and (not gaussian_usdz.is_file() or gaussian_usdz.suffix.lower() != ".usdz"):
        parser.error(f"Gaussian scene must be an existing .usdz file: {gaussian_usdz}")

    print("Loading .ply mesh...")
    vertices, faces, normals, colors = load_ply_mesh(source)
    print(f"Loaded {len(vertices):,} vertices and {len(faces):,} triangles from {source}", flush=True)
    output = write_usd(vertices, faces, normals, colors, output, args.scale, args.up_axis, args.meters_per_unit)
    print(f"Wrote mesh asset {output}")
    if gaussian_usdz:
        scene_output = (
            args.scene_output.expanduser().resolve()
            if args.scene_output
            else output.with_name(f"{output.stem}_with_gaussians.usda")
        )
        if scene_output.suffix.lower() != ".usda":
            parser.error("--scene-output must have a .usda extension")
        if scene_output in (output, gaussian_usdz):
            parser.error("composed USDA path must differ from both input/output assets")
        scene_output = write_combined_scene(
            gaussian_usdz, output, scene_output, args.gaussian_prim
        )
        print(f"Wrote combined scene {scene_output}")


if __name__ == "__main__":
    main()
