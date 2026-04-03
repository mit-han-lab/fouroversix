module scale_factors_multiplier (
    input  logic [7:0]  sf_w_fp8,
    input  logic [7:0]  sf_a_fp8,

    output logic        is_int4_w,
    output logic        is_int4_a,
    output logic [31:0] sf_fp32
);

localparam logic [7:0]  FP32_BIAS = 8'd127;
localparam logic [31:0] ONE_FP32 = 32'h3F800000;
localparam logic [31:0] SIX_SEVENTHS_FP32 = 32'h3F5B6DB7;
localparam logic [31:0] THIRTY_SIX_FORTY_NINTHS_FP32 = 32'h3F3C14E6;

// ------------------------
// Internal signals
// ------------------------
wire [3:0] sf_w_fp8_exp = sf_w_fp8[6:3];
wire [2:0] sf_w_fp8_man = sf_w_fp8[2:0];
wire [3:0] sf_a_fp8_exp = sf_a_fp8[6:3];
wire [2:0] sf_a_fp8_man = sf_a_fp8[2:0];

assign is_int4_w = sf_w_fp8[7];
assign is_int4_a = sf_a_fp8[7];


// ------------------------
// 0. Handle zero input case
// ------------------------
wire is_input_zero = ((sf_w_fp8_exp == 4'd0) && (sf_w_fp8_man == 3'd0)) || ((sf_a_fp8_exp == 4'd0) && (sf_a_fp8_man == 3'd0));

// ------------------------
// 1. FP8 decoding
// ------------------------
// Match the existing UE4M3 decode style:
// - exp==0 -> subnormal, effective exponent is 1 and hidden bit is 0
// - exp!=0 -> normal, effective exponent is exp and hidden bit is 1
// - exp==1111 && man==111 is treated as max finite by clamping man to 110
logic [3:0] sf_w_exp;
logic       sf_w_man_hid;
logic [2:0] sf_w_man_in;
logic [3:0] sf_w_man_uq13;

logic [3:0] sf_a_exp;
logic       sf_a_man_hid;
logic [2:0] sf_a_man_in;
logic [3:0] sf_a_man_uq13;

assign sf_w_man_uq13 = {sf_w_man_hid, sf_w_man_in};
assign sf_a_man_uq13 = {sf_a_man_hid, sf_a_man_in};

always_comb begin
    if (sf_w_fp8_exp == 4'd0) begin
        sf_w_exp = 4'd1;
        sf_w_man_hid = 1'b0;
        sf_w_man_in = sf_w_fp8_man;
    end else begin
        sf_w_exp = sf_w_fp8_exp;
        sf_w_man_hid = 1'b1;
        sf_w_man_in = (sf_w_fp8_exp == 4'b1111 && sf_w_fp8_man == 3'b111) ? 3'b110 : sf_w_fp8_man;
    end

    if (sf_a_fp8_exp == 4'd0) begin
        sf_a_exp = 4'd1;
        sf_a_man_hid = 1'b0;
        sf_a_man_in = sf_a_fp8_man;
    end else begin
        sf_a_exp = sf_a_fp8_exp;
        sf_a_man_hid = 1'b1;
        sf_a_man_in = (sf_a_fp8_exp == 4'b1111 && sf_a_fp8_man == 3'b111) ? 3'b110 : sf_a_fp8_man;
    end
end


// ------------------------
// 2. Multiply the decoded FP8 values and express the raw product in FP32
// ------------------------
// Q1.3 * Q1.3 -> Q2.6
logic [7:0] sf_p_uq26;
assign sf_p_uq26 = sf_w_man_uq13 * sf_a_man_uq13;

// value = sf_p_uq26 * 2^(exp_eff), where exp_eff = w_exp + a_exp - 20
// because each Q1.3 operand contributes a factor of 2^-3 and FP8 bias is 7.
logic signed [5:0] exp_eff;
assign exp_eff = $signed({1'b0, sf_w_exp}) + $signed({1'b0, sf_a_exp}) - 6'sd20;

logic        sf_y_sig;
logic [7:0]  sf_y_exp;
logic [22:0] sf_y_man;
logic [31:0] sf_raw_fp32;
logic [29:0] sf_p_norm_uq023;
logic [3:0]  sf_p_uq26_left1_pos;
logic signed [5:0] sf_y_exp_unbiased;
logic signed [8:0] sf_y_exp_biased;

// Product bounds:
// - min non-zero raw product = 2^-18
// - max raw product = 448 * 448 = 200704
// After optional 6/7 or 36/49 scaling:
// - min non-zero final product = (36/49) * 2^-18 ~= 2.80e-6
// - max final product = 200704
// These are all normal finite FP32 values, so the datapath never needs
// FP32 subnormal, NaN, or infinity handling.
assign sf_raw_fp32 = {sf_y_sig, sf_y_exp, sf_y_man};
assign sf_y_sig = 1'b0;

always_comb begin
    unique casez (sf_p_uq26)
        8'b1???????: sf_p_uq26_left1_pos = 4'd7;
        8'b01??????: sf_p_uq26_left1_pos = 4'd6;
        8'b001?????: sf_p_uq26_left1_pos = 4'd5;
        8'b0001????: sf_p_uq26_left1_pos = 4'd4;
        8'b00001???: sf_p_uq26_left1_pos = 4'd3;
        8'b000001??: sf_p_uq26_left1_pos = 4'd2;
        8'b0000001?: sf_p_uq26_left1_pos = 4'd1;
        8'b00000001: sf_p_uq26_left1_pos = 4'd0;
        default:     sf_p_uq26_left1_pos = 4'd0;
    endcase

    if (sf_p_uq26 == 8'd0) begin
        sf_y_exp_unbiased = 6'sd0;
        sf_y_exp_biased = 9'sd0;
        sf_y_exp = 8'd0;
        sf_y_man = 23'd0;
        sf_p_norm_uq023 = 30'd0;
    end else begin
        sf_y_exp_unbiased = exp_eff + $signed({2'b00, sf_p_uq26_left1_pos});
        sf_y_exp_biased = $signed({1'b0, FP32_BIAS}) + sf_y_exp_unbiased;
        sf_y_exp = sf_y_exp_biased[7:0];
        sf_p_norm_uq023 = ({22'd0, sf_p_uq26} << (23 - sf_p_uq26_left1_pos));
        sf_y_man = sf_p_norm_uq023[22:0];
    end
end


// ------------------------
// 3. Apply optional range-alignment scaling in FP32
// ------------------------
logic        apply_scaling;
logic [31:0] sf_scale_factor_fp32;
logic [31:0] sf_scaled_fp32;

assign apply_scaling = is_int4_w || is_int4_a;
assign sf_scale_factor_fp32 =
    (is_int4_w && is_int4_a) ? THIRTY_SIX_FORTY_NINTHS_FP32 :
    (is_int4_w || is_int4_a) ? SIX_SEVENTHS_FP32 :
                               ONE_FP32;

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
) sf_scale_mult_inst (
    .a(sf_raw_fp32),
    .b(sf_scale_factor_fp32),
    .rnd(3'b000),
    .z(sf_scaled_fp32),
    .status()
);

assign sf_fp32 = is_input_zero ? 32'd0 : (apply_scaling ? sf_scaled_fp32 : sf_raw_fp32);

endmodule // scale_factors_multiplier
