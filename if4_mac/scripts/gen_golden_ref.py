#!/usr/bin/env python3

import random
import struct
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
TB_DIR = SCRIPT_DIR.parent / "tb"

BLOCK_SIZE = 16
SEGMENT_LENGTH = 10
NUM_EXPECTED = 3
NUM_INPUTS = (SEGMENT_LENGTH * NUM_EXPECTED) + 1
IF4_RANDOM_SEED = 20260402
NVFP4_RANDOM_SEED = 20260403

VALID_4BIT = [0x0, 0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0x9, 0xA, 0xB, 0xC, 0xD, 0xE, 0xF]
SIX_SEVENTHS_F32 = struct.unpack(">f", struct.pack(">f", 6.0 / 7.0))[0]
THIRTY_SIX_FORTY_NINTHS_F32 = struct.unpack(">f", struct.pack(">f", 36.0 / 49.0))[0]


def f32(value: float) -> float:
    return struct.unpack(">f", struct.pack(">f", float(value)))[0]


def f16(value: float) -> float:
    return struct.unpack(">e", struct.pack(">e", float(value)))[0]


def f32_hex(value: float) -> str:
    bits = struct.unpack(">I", struct.pack(">f", f32(value)))[0]
    return f"32'h{bits:08X}"


def decode_nvfp4(bits: int) -> float:
    if bits == 0x8:
        raise ValueError("NVFP4 -0.0 is invalid input")

    sign = (bits >> 3) & 0x1
    mag = {
        0x0: 0.0,
        0x1: 0.5,
        0x2: 1.0,
        0x3: 1.5,
        0x4: 2.0,
        0x5: 3.0,
        0x6: 4.0,
        0x7: 6.0,
    }[bits & 0x7]
    return -mag if sign else mag


def decode_int4(bits: int) -> float:
    if bits == 0x8:
        raise ValueError("INT4 -8 is invalid input")

    sign = (bits >> 3) & 0x1
    val = bits & 0x7
    signed = val if sign == 0 else (val - 8)
    return float(signed)


def decode_if4(bits: int, is_int4: int) -> float:
    return decode_int4(bits) if is_int4 else decode_nvfp4(bits)


def decode_fp8_payload(bits: int) -> float:
    exp = (bits >> 3) & 0xF
    man = bits & 0x7
    bias = 7

    if exp == 0:
        return (man / 8.0) * (2.0 ** (1 - bias))

    if exp == 0xF and man == 0x7:
        man = 0x6

    return (1.0 + man / 8.0) * (2.0 ** (exp - bias))


def pairwise_tree_sum(values: list[float]) -> float:
    stage = [f32(value) for value in values]
    while len(stage) > 1:
        next_stage = []
        idx = 0
        while idx + 1 < len(stage):
            next_stage.append(f32(stage[idx] + stage[idx + 1]))
            idx += 2
        if idx < len(stage):
            next_stage.append(stage[idx])
        stage = next_stage
    return stage[0]


def make_lanes(rng: random.Random) -> list[int]:
    lanes = []
    for _ in range(BLOCK_SIZE):
        lanes.append(rng.choice(VALID_4BIT))
    return lanes


def random_if4_sf(rng: random.Random) -> int:
    return (rng.randrange(2) << 7) | rng.randrange(0x80)


def random_nvfp4_sf(rng: random.Random) -> int:
    return rng.randrange(0x80)


def build_segment(
    *,
    rng: random.Random,
    sf_generator,
) -> list[dict]:
    segment = []
    for cycle_idx in range(SEGMENT_LENGTH):
        item = {
            "first": 1 if cycle_idx == 0 else 0,
            "a_sf": sf_generator(rng),
            "w_sf": sf_generator(rng),
            "a_lane": make_lanes(rng),
            "w_lane": make_lanes(rng),
        }
        segment.append(item)
    return segment


def build_if4_mac_inputs() -> list[dict]:
    rng = random.Random(IF4_RANDOM_SEED)
    inputs = []
    for _ in range(NUM_EXPECTED):
        inputs.extend(build_segment(rng=rng, sf_generator=random_if4_sf))
    inputs.append(
        {
            "first": 1,
            "a_sf": 0x00,
            "w_sf": 0x00,
            "a_lane": [0] * BLOCK_SIZE,
            "w_lane": [0] * BLOCK_SIZE,
        }
    )
    return inputs


def build_nvfp4_mac_inputs() -> list[dict]:
    rng = random.Random(NVFP4_RANDOM_SEED)
    inputs = []
    for _ in range(NUM_EXPECTED):
        inputs.extend(build_segment(rng=rng, sf_generator=random_nvfp4_sf))
    inputs.append(
        {
            "first": 1,
            "a_sf": 0x00,
            "w_sf": 0x00,
            "a_lane": [0] * BLOCK_SIZE,
            "w_lane": [0] * BLOCK_SIZE,
        }
    )
    return inputs


def generate_if4_mac_golden_ref(inputs: list[dict]) -> list[str]:
    expected = []
    acc_by_lane = [f32(0.0)] * BLOCK_SIZE
    for index, item in enumerate(inputs):
        if item["first"] and index != 0:
            expected.append(f32_hex(pairwise_tree_sum(acc_by_lane)))
            acc_by_lane = [f32(0.0)] * BLOCK_SIZE

        a_is_int4 = (item["a_sf"] >> 7) & 0x1
        w_is_int4 = (item["w_sf"] >> 7) & 0x1
        a_scale = decode_fp8_payload(item["a_sf"] & 0x7F)
        w_scale = decode_fp8_payload(item["w_sf"] & 0x7F)
        sf_fp32 = f32(a_scale * w_scale)
        if a_is_int4 and w_is_int4:
            sf_fp32 = f32(sf_fp32 * THIRTY_SIX_FORTY_NINTHS_F32)
        elif a_is_int4 or w_is_int4:
            sf_fp32 = f32(sf_fp32 * SIX_SEVENTHS_F32)

        for lane in range(BLOCK_SIZE):
            a_value = decode_if4(item["a_lane"][lane], a_is_int4)
            w_value = decode_if4(item["w_lane"][lane], w_is_int4)
            wa_fp16 = f16(a_value * w_value)
            mul_fp32 = f32(wa_fp16 * sf_fp32)
            acc_by_lane[lane] = f32(acc_by_lane[lane] + mul_fp32)

    return expected


def generate_nvfp4_mac_golden_ref(inputs: list[dict]) -> list[str]:
    expected = []
    acc_by_lane = [f32(0.0)] * BLOCK_SIZE
    for index, item in enumerate(inputs):
        if item["first"] and index != 0:
            expected.append(f32_hex(pairwise_tree_sum(acc_by_lane)))
            acc_by_lane = [f32(0.0)] * BLOCK_SIZE

        a_scale = decode_fp8_payload(item["a_sf"] & 0x7F)
        w_scale = decode_fp8_payload(item["w_sf"] & 0x7F)
        sf_fp32 = f32(a_scale * w_scale)

        for lane in range(BLOCK_SIZE):
            a_value = decode_nvfp4(item["a_lane"][lane])
            w_value = decode_nvfp4(item["w_lane"][lane])
            wa_fp16 = f16(a_value * w_value)
            mul_fp32 = f32(wa_fp16 * sf_fp32)
            acc_by_lane[lane] = f32(acc_by_lane[lane] + mul_fp32)

    return expected


def emit_include(path: Path, inputs: list[dict], expected_hex: list[str]) -> None:
    lines = []
    lines.append("// Generated by scripts/gen_golden_ref.py")
    lines.append("")

    for idx, item in enumerate(inputs):
        lines.append(f"  first_vec[{idx}] = 1'b{item['first']};")
        lines.append(f"  a_sf_vec[{idx}] = 8'h{item['a_sf']:02X};")
        lines.append(f"  w_sf_vec[{idx}] = 8'h{item['w_sf']:02X};")
        for lane in range(BLOCK_SIZE):
            lines.append(f"  a_vec[{idx}][{lane}] = 4'h{item['a_lane'][lane]:X};")
            lines.append(f"  w_vec[{idx}][{lane}] = 4'h{item['w_lane'][lane]:X};")
        lines.append("")

    for idx, acc_hex in enumerate(expected_hex):
        lines.append(f"  expected_acc[{idx}] = {acc_hex};")

    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def main() -> None:
    TB_DIR.mkdir(parents=True, exist_ok=True)

    if4_inputs = build_if4_mac_inputs()
    nvfp4_inputs = build_nvfp4_mac_inputs()

    if4_expected = generate_if4_mac_golden_ref(if4_inputs)
    nvfp4_expected = generate_nvfp4_mac_golden_ref(nvfp4_inputs)

    if len(if4_expected) != NUM_EXPECTED:
        raise RuntimeError("IF4 expected output count does not match NUM_EXPECTED")
    if len(nvfp4_expected) != NUM_EXPECTED:
        raise RuntimeError("NVFP4 expected output count does not match NUM_EXPECTED")

    emit_include(TB_DIR / "if4_mac_golden_ref.svh", if4_inputs, if4_expected)
    emit_include(TB_DIR / "nvfp4_mac_golden_ref.svh", nvfp4_inputs, nvfp4_expected)


if __name__ == "__main__":
    main()
