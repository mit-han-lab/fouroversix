module w_a_multiplier (
    input  logic [3:0]  w_nvfp4,
    input  logic [3:0]  a_nvfp4,

    output logic [15:0] wa_fp16
);

localparam int FP16_BIAS = 15;

wire       w_sig = w_nvfp4[3];
wire [2:0] w_val = w_nvfp4[2:0];
wire       a_sig = a_nvfp4[3];
wire [2:0] a_val = a_nvfp4[2:0];

// In NVFP4, xxx_val==0 means +/-0 and 4'b1000 is invalid (-0).
wire w_is_zero = (w_val == 3'd0);
wire a_is_zero = (a_val == 3'd0);

function automatic logic [3:0] decode_nvfp4_uq31 (
    input logic [3:0] nvfp4_bits
);
    logic [2:0] val;
    begin
        val = nvfp4_bits[2:0];

        unique case (val)
            3'd0: decode_nvfp4_uq31 = 4'b0000; // 0.0
            3'd1: decode_nvfp4_uq31 = 4'b0001; // 0.5
            3'd2: decode_nvfp4_uq31 = 4'b0010; // 1.0
            3'd3: decode_nvfp4_uq31 = 4'b0011; // 1.5
            3'd4: decode_nvfp4_uq31 = 4'b0100; // 2.0
            3'd5: decode_nvfp4_uq31 = 4'b0110; // 3.0
            3'd6: decode_nvfp4_uq31 = 4'b1000; // 4.0
            3'd7: decode_nvfp4_uq31 = 4'b1100; // 6.0
            default: decode_nvfp4_uq31 = 4'd0;
        endcase
    end
endfunction

always_comb begin
    assert (!(w_nvfp4 === 4'b1000))
        else $error("w_a_multiplier: invalid NVFP4 w_nvfp4=-0");
    assert (!(a_nvfp4 === 4'b1000))
        else $error("w_a_multiplier: invalid NVFP4 a_nvfp4=-0");
end

// ------------------------
// 0. Wire output
// ------------------------
logic [15:0] wa_fp16_calc;
assign wa_fp16 = (w_is_zero || a_is_zero) ? {(w_sig ^ a_sig), 15'd0} : wa_fp16_calc;

// ------------------------
// 1. IF4 decode
// ------------------------
// Represent both operands as unsigned Q3.1 magnitudes using the NVFP4 value map.
logic [3:0] w_uq31;
logic [3:0] a_uq31;
assign w_uq31 = decode_nvfp4_uq31(w_nvfp4);
assign a_uq31 = decode_nvfp4_uq31(a_nvfp4);

// ------------------------
// 2. Multiply magnitudes
// ------------------------
// Q3.1 * Q3.1 -> Q6.2. Valid non-zero products are in [0.25, 36].
logic [7:0] p_uq62;
logic       p_sig;
assign p_uq62 = w_uq31 * a_uq31;
assign p_sig  = w_sig ^ a_sig;

// ------------------------
// 3. Express in FP16
// ------------------------
// FP16 format: 1 sign bit, 5 exponent bits with bias 15, 10 mantissa bits.
// Product bounds guarantee:
// - no NaN/Inf: max finite result is 36 << 65504
// - no subnormal: min non-zero result is 0.25 >> 2^-14
logic        y_sig;
logic [4:0]  y_exp;
logic [9:0]  y_man;
logic [17:0] p_norm_uq010;
logic [3:0]  p_uq62_left1_pos;
logic signed [5:0] y_exp_unbiased;
logic signed [6:0] y_exp_biased;

assign wa_fp16_calc = {y_sig, y_exp, y_man};
assign y_sig = p_sig;

always_comb begin
    unique casez (p_uq62)
        8'b1???????: p_uq62_left1_pos = 4'd7;
        8'b01??????: p_uq62_left1_pos = 4'd6;
        8'b001?????: p_uq62_left1_pos = 4'd5;
        8'b0001????: p_uq62_left1_pos = 4'd4;
        8'b00001???: p_uq62_left1_pos = 4'd3;
        8'b000001??: p_uq62_left1_pos = 4'd2;
        8'b0000001?: p_uq62_left1_pos = 4'd1;
        8'b00000001: p_uq62_left1_pos = 4'd0;
        default:        p_uq62_left1_pos = 4'd0;
    endcase

    if (p_uq62 == 8'd0) begin
        y_exp_unbiased = 6'sd0;
        y_exp_biased = 7'sd0;
        y_exp = 5'd0;
        y_man = 10'd0;
        p_norm_uq010 = 18'd0;
    end else begin
        y_exp_unbiased = $signed({2'b00, p_uq62_left1_pos}) - 6'sd2;
        y_exp_biased = $signed({1'b0, FP16_BIAS}) + y_exp_unbiased;
        y_exp = y_exp_biased[4:0];
        p_norm_uq010 = ({10'd0, p_uq62} << (10 - p_uq62_left1_pos));
        y_man = p_norm_uq010[9:0];
    end
end


endmodule // w_a_multiplier
