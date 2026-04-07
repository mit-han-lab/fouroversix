#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 {if4|nvfp4}" >&2
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TB_DIR="${ROOT_DIR}/tb"
RTL_DIR="${ROOT_DIR}/rtl"

: "${CW_SIM_DIR:?Set CW_SIM_DIR to the directory containing CW_fp_add.v and CW_fp_mult.v}"

case "$1" in
  if4)
    TOP=tb_if4_mac
    RTL_SUBDIR="${RTL_DIR}/if4_mac"
    TB_FILE="${TB_DIR}/tb_if4_mac.sv"
    ;;
  nvfp4)
    TOP=tb_nvfp4_mac
    RTL_SUBDIR="${RTL_DIR}/nvfp4_mac"
    TB_FILE="${TB_DIR}/tb_nvfp4_mac.sv"
    ;;
  *)
    echo "Unknown target: $1" >&2
    exit 1
    ;;
esac

xrun \
  -clean \
  -sv \
  -timescale 1ns/1ps \
  -top "${TOP}" \
  -incdir "${TB_DIR}" \
  "${CW_SIM_DIR}/CW_fp_mult.v" \
  "${CW_SIM_DIR}/CW_fp_add.v" \
  "${RTL_SUBDIR}/w_a_multiplier.sv" \
  "${RTL_SUBDIR}/scale_factors_multiplier.sv" \
  "${RTL_SUBDIR}/wa_sf_multiplier.sv" \
  "${RTL_SUBDIR}/$(basename "${RTL_SUBDIR}").sv" \
  "${TB_FILE}"
