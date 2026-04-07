module wa_sf_multiplier (
    input  logic [15:0] wa_fp16,
    input  logic [31:0] sf_fp32,

    output logic [31:0] mul_fp32
);

localparam logic [7:0] FP32_EXP_FROM_FP16_ZERO = 8'd112; // 127 - 15

// ------------------------
// Internal signals
// ------------------------
wire        wa_sig = wa_fp16[15];
wire [4:0]  wa_exp = wa_fp16[14:10];
wire [9:0]  wa_man = wa_fp16[9:0];
wire        wa_is_zero = (wa_exp == 5'd0) && (wa_man == 10'd0);


// ------------------------
// 0. Convert wa_fp16 to FP32
// ------------------------
// w_a_multiplier guarantees:
// - zero or normal finite FP16 only
// - no FP16 subnormal, NaN, or Inf cases
// So FP16 -> FP32 is just sign copy, exponent rebias, and mantissa extension.
logic [31:0] wa_fp32;
assign wa_fp32 =
    wa_is_zero ? {wa_sig, 31'd0} :
                 {wa_sig, wa_exp + FP32_EXP_FROM_FP16_ZERO, wa_man, 13'd0};


// ------------------------
// 1. Multiply wa_fp32 and sf_fp32
// ------------------------
// Product bounds remain well within FP32 normal finite range:
// - min non-zero = 0.25 * ((36/49) * 2^-18) ~= 7.01e-7
// - max          = 49 * 200704 = 9834496
localparam MANTISSA_BW = 23;
localparam EXPONENT_BW = 8;
localparam IEEE_COMPLIANCE = 0;
localparam ARCH = 1;
localparam RETURN_SIGNED_NANS = 0;

CW_fp_mult #(
    MANTISSA_BW,
    EXPONENT_BW,
    IEEE_COMPLIANCE,
    ARCH,
    RETURN_SIGNED_NANS
) mul_fp32_inst (
    .a(wa_fp32),
    .b(sf_fp32),
    .rnd(3'b000),
    .z(mul_fp32),
    .status()
);

endmodule // wa_sf_multiplier
