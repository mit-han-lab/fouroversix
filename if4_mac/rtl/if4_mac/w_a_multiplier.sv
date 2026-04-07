module w_a_multiplier (
    input  logic [3:0]  w_if4,
    input  logic [3:0]  a_if4,

    input  logic        is_int4_w,
    input  logic        is_int4_a,

	output logic [15:0] wa_fp16
);

localparam int FP16_BIAS = 15;

wire       w_sig = w_if4[3];
wire [2:0] w_val = w_if4[2:0];
wire       a_sig = a_if4[3];
wire [2:0] a_val = a_if4[2:0];

// In NVFP4 mode, xxx_val==0 means +/-0.
// In INT4 mode, only 4'b0000 is zero; 4'b1000 is invalid for this block.
wire w_is_zero = is_int4_w ? (w_if4 == 4'b0000) : (w_val == 3'd0);
wire a_is_zero = is_int4_a ? (a_if4 == 4'b0000) : (a_val == 3'd0);

function automatic logic [4:0] decode_if4_uq41 (
    input logic [3:0] if4_bits,
    input logic       is_int4
);
    logic       sig;
    logic [2:0] val;
    logic [3:0] int_mag;
    begin
        sig = if4_bits[3];
        val = if4_bits[2:0];

        if (is_int4) begin
            int_mag = sig ? (4'd8 - {1'b0, val}) : {1'b0, val};
            decode_if4_uq41 = {int_mag, 1'b0};
        end else begin
            unique case (val)
                3'd0: decode_if4_uq41 = 5'b00000; // 0.0
                3'd1: decode_if4_uq41 = 5'b00001; // 0.5
                3'd2: decode_if4_uq41 = 5'b00010; // 1.0
                3'd3: decode_if4_uq41 = 5'b00011; // 1.5
                3'd4: decode_if4_uq41 = 5'b00100; // 2.0
                3'd5: decode_if4_uq41 = 5'b00110; // 3.0
                3'd6: decode_if4_uq41 = 5'b01000; // 4.0
                3'd7: decode_if4_uq41 = 5'b01100; // 6.0
                default: decode_if4_uq41 = 5'd0;
            endcase
        end
    end
endfunction

always_comb begin
    assert (!(is_int4_w && (w_if4 === 4'b1000)))
        else $error("w_a_multiplier: invalid int4 w_if4=-8");
    assert (!(is_int4_a && (a_if4 === 4'b1000)))
        else $error("w_a_multiplier: invalid int4 a_if4=-8");
    assert (!(!is_int4_w && (w_if4 === 4'b1000)))
        else $error("w_a_multiplier: invalid NVFP4 w_if4=-0");
    assert (!(!is_int4_a && (a_if4 === 4'b1000)))
        else $error("w_a_multiplier: invalid NVFP4 a_if4=-0");
end

// ------------------------
// 0. Wire output
// ------------------------
logic [15:0] wa_fp16_calc;
assign wa_fp16 = (w_is_zero || a_is_zero) ? {(w_sig ^ a_sig), 15'd0} : wa_fp16_calc;

// ------------------------
// 1. IF4 decode
// ------------------------
// Represent both operands as unsigned Q4.1 magnitudes.
// Valid INT4 inputs are -7..7, so their magnitudes are 0..7.
logic [4:0] w_uq41;
logic [4:0] a_uq41;
assign w_uq41 = decode_if4_uq41(w_if4, is_int4_w);
assign a_uq41 = decode_if4_uq41(a_if4, is_int4_a);

// ------------------------
// 2. Multiply magnitudes
// ------------------------
// Q4.1 * Q4.1 -> Q8.2. Valid non-zero products are in [0.25, 49].
logic [9:0] p_uq82;
logic       p_sig;
assign p_uq82 = w_uq41 * a_uq41;
assign p_sig  = w_sig ^ a_sig;

// ------------------------
// 3. Express in FP16
// ------------------------
// FP16 format: 1 sign bit, 5 exponent bits with bias 15, 10 mantissa bits.
// Product bounds guarantee:
// - no NaN/Inf: max finite result is 49 << 65504
// - no subnormal: min non-zero result is 0.25 >> 2^-14
logic        y_sig;
logic [4:0]  y_exp;
logic [9:0]  y_man;
logic [19:0] p_norm_uq010;
logic [3:0]  p_uq82_left1_pos;
logic signed [5:0] y_exp_unbiased;
logic signed [6:0] y_exp_biased;

assign wa_fp16_calc = {y_sig, y_exp, y_man};
assign y_sig = p_sig;

always_comb begin
    unique casez (p_uq82)
        10'b1?????????: p_uq82_left1_pos = 4'd9;
        10'b01????????: p_uq82_left1_pos = 4'd8;
        10'b001???????: p_uq82_left1_pos = 4'd7;
        10'b0001??????: p_uq82_left1_pos = 4'd6;
        10'b00001?????: p_uq82_left1_pos = 4'd5;
        10'b000001????: p_uq82_left1_pos = 4'd4;
        10'b0000001???: p_uq82_left1_pos = 4'd3;
        10'b00000001??: p_uq82_left1_pos = 4'd2;
        10'b000000001?: p_uq82_left1_pos = 4'd1;
        10'b0000000001: p_uq82_left1_pos = 4'd0;
        default:        p_uq82_left1_pos = 4'd0;
    endcase

    if (p_uq82 == 10'd0) begin
        y_exp_unbiased = 6'sd0;
        y_exp_biased = 7'sd0;
        y_exp = 5'd0;
        y_man = 10'd0;
        p_norm_uq010 = 20'd0;
    end else begin
        y_exp_unbiased = $signed({2'b00, p_uq82_left1_pos}) - 6'sd2;
        y_exp_biased = $signed({1'b0, FP16_BIAS}) + y_exp_unbiased;
        y_exp = y_exp_biased[4:0];
        p_norm_uq010 = ({10'd0, p_uq82} << (10 - p_uq82_left1_pos));
        y_man = p_norm_uq010[9:0];
    end
end


endmodule // w_a_multiplier
