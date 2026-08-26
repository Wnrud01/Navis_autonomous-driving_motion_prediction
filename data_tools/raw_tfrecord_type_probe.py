#!/usr/bin/env python3
"""Read-only TF Example parser for sampling agent type codes from TFRecord files.

No TensorFlow dependency is required. The parser only extracts state/type and
current size/valid fields from standard tf.train.Example records.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import struct
from collections import Counter, defaultdict
from typing import Iterator


def read_varint(buf: bytes, pos: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while True:
        if pos >= len(buf):
            raise ValueError("Truncated varint")
        byte = buf[pos]
        pos += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, pos
        shift += 7
        if shift > 70:
            raise ValueError("Malformed varint")


def iter_fields(buf: bytes) -> Iterator[tuple[int, int, bytes | int]]:
    pos = 0
    while pos < len(buf):
        tag, pos = read_varint(buf, pos)
        number, wire = tag >> 3, tag & 0x07
        if wire == 0:
            value, pos = read_varint(buf, pos)
        elif wire == 1:
            value, pos = buf[pos : pos + 8], pos + 8
        elif wire == 2:
            size, pos = read_varint(buf, pos)
            value, pos = buf[pos : pos + size], pos + size
        elif wire == 5:
            value, pos = buf[pos : pos + 4], pos + 4
        else:
            raise ValueError(f"Unsupported wire type {wire}")
        yield number, wire, value


def parse_packed_varints(buf: bytes) -> list[int]:
    values: list[int] = []
    pos = 0
    while pos < len(buf):
        value, pos = read_varint(buf, pos)
        values.append(value)
    return values


def parse_feature(feature: bytes) -> tuple[str, list[int] | list[float] | list[bytes]]:
    for number, wire, value in iter_fields(feature):
        if wire != 2:
            continue
        nested = value if isinstance(value, bytes) else b""
        if number == 1:  # BytesList
            items = [v for n, w, v in iter_fields(nested) if n == 1 and w == 2]
            return "bytes", [v for v in items if isinstance(v, bytes)]
        if number == 2:  # FloatList
            items: list[float] = []
            for n, w, v in iter_fields(nested):
                if n == 1 and w == 2 and isinstance(v, bytes):
                    items.extend(struct.unpack("<" + "f" * (len(v) // 4), v))
            return "float", items
        if number == 3:  # Int64List
            items: list[int] = []
            for n, w, v in iter_fields(nested):
                if n == 1 and w == 2 and isinstance(v, bytes):
                    items.extend(parse_packed_varints(v))
                elif n == 1 and w == 0 and isinstance(v, int):
                    items.append(v)
            return "int", items
    return "none", []


def parse_example(record: bytes) -> dict[str, tuple[str, list[int] | list[float] | list[bytes]]]:
    features_message = b""
    for number, wire, value in iter_fields(record):
        if number == 1 and wire == 2 and isinstance(value, bytes):
            features_message = value
            break
    result = {}
    for number, wire, entry in iter_fields(features_message):
        if number != 1 or wire != 2 or not isinstance(entry, bytes):
            continue
        key = None
        feature = None
        for en, ew, ev in iter_fields(entry):
            if en == 1 and ew == 2 and isinstance(ev, bytes):
                key = ev.decode("utf-8", errors="replace")
            elif en == 2 and ew == 2 and isinstance(ev, bytes):
                feature = ev
        if key is not None and feature is not None:
            result[key] = parse_feature(feature)
    return result


def first_record(path: str) -> bytes:
    with open(path, "rb") as handle:
        header = handle.read(12)
        if len(header) != 12:
            raise ValueError("TFRecord header is too short")
        length = struct.unpack("<Q", header[:8])[0]
        payload = handle.read(length)
        if len(payload) != length:
            raise ValueError("Truncated TFRecord payload")
        return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--max-files", type=int, default=200)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    paths = sorted(glob.glob(os.path.join(args.data_dir, "**", "*.tfrecord"), recursive=True))[: args.max_files]
    if not paths:
        raise FileNotFoundError("No TFRecord files found")

    type_counts: Counter[int] = Counter()
    valid_counts: Counter[int] = Counter()
    dim_sums: defaultdict[int, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])
    errors: list[str] = []
    used = 0

    for path in paths:
        try:
            example = parse_example(first_record(path))
            type_kind, types = example.get("state/type", ("none", []))
            valid_kind, valid = example.get("state/current/valid", ("none", []))
            length_kind, lengths = example.get("state/current/length", ("none", []))
            width_kind, widths = example.get("state/current/width", ("none", []))
            height_kind, heights = example.get("state/current/height", ("none", []))
            if type_kind != "int" or not types:
                raise ValueError("state/type missing or not Int64List")
            used += 1
            for i, raw_type in enumerate(types):
                code = int(raw_type)
                is_valid = int(valid[i]) if valid_kind == "int" and i < len(valid) else 1
                type_counts[code] += 1
                valid_counts[code] += is_valid
                if is_valid and length_kind == width_kind == height_kind == "float" and i < len(lengths) and i < len(widths) and i < len(heights):
                    dim_sums[code][0] += float(lengths[i])
                    dim_sums[code][1] += float(widths[i])
                    dim_sums[code][2] += float(heights[i])
                    dim_sums[code][3] += 1.0
        except Exception as exc:
            errors.append(f"{os.path.basename(path)}: {exc}")

    codes = {}
    for code in sorted(type_counts):
        total = type_counts[code]
        n_dim = dim_sums[code][3]
        codes[str(code)] = {
            "all_agents": total,
            "current_valid_agents": valid_counts[code],
            "current_valid_ratio": valid_counts[code] / total if total else 0.0,
            "mean_length_m_if_current_valid": dim_sums[code][0] / n_dim if n_dim else None,
            "mean_width_m_if_current_valid": dim_sums[code][1] / n_dim if n_dim else None,
            "mean_height_m_if_current_valid": dim_sums[code][2] / n_dim if n_dim else None,
        }
    result = {
        "sampled_files": len(paths),
        "parsed_files": used,
        "failed_files": len(errors),
        "type_codes": codes,
        "errors_sample": errors[:10],
        "note": "Codes are reported as stored. Semantic labels require the dataset type-code mapping.",
    }
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
