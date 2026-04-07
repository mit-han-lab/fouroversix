module if4_mac #(
    parameter BLOCK_SIZE = 16
) (
    input  logic        clk,
    input  logic        rst_n,
    input  logic        en,
    input  logic        first,

    input  logic [3:0]  a_in [BLOCK_SIZE-1:0],
    input  logic [7:0]  a_sf,
    input  logic [3:0]  w_in [BLOCK_SIZE-1:0],
    input  logic [7:0]  w_sf,

    output logic [31:0] acc,
    output logic        done
);

localparam MANTISSA_BW = 23;
localparam EXPONENT_BW = 8;
localparam IEEE_COMPLIANCE = 0;
localparam ARCH = 1;
localparam IEEE_NAN_COMPLIANCE = 0;
localparam int NUM_FP_ADD = BLOCK_SIZE / 2;
localparam int REDUCE_CNT_W = $clog2(BLOCK_SIZE + 1);

typedef enum logic [0:0] {
    REDUCE_IDLE,
    REDUCE_STEP
} reduce_state_t;

// ------------------------
// Internal signals
// ------------------------
logic        is_int4_w;
logic        is_int4_a;
logic [15:0] wa_fp16_comb [BLOCK_SIZE-1:0];
logic [31:0] sf_fp32_comb;
logic [31:0] mul_comb [BLOCK_SIZE-1:0];

logic [31:0] mul_stage1 [BLOCK_SIZE-1:0];
logic        stage1_valid;
logic        first_stage1;

logic [31:0] acc_lane [BLOCK_SIZE-1:0];
logic [31:0] acc_lane_add_out [BLOCK_SIZE-1:0];
logic [7:0]  acc_lane_add_status [BLOCK_SIZE-1:0];
logic        batch_active;

logic        snapshot_valid;
logic [31:0] snapshot_vals [BLOCK_SIZE-1:0];

reduce_state_t reduce_state;
logic [REDUCE_CNT_W-1:0] reduce_count;
logic [REDUCE_CNT_W-1:0] pair_count;
logic [REDUCE_CNT_W-1:0] next_count;
logic [31:0]             reduce_vals [BLOCK_SIZE-1:0];
logic [31:0]             reduce_add_out [NUM_FP_ADD-1:0];
logic [7:0]              reduce_add_status [NUM_FP_ADD-1:0];

// ------------------------
// Instance submodules
// ------------------------
// One shared scale-factor product is broadcast to every lane. Each lane forms
// its own wa_fp16, then multiplies that by the shared sf_fp32.

scale_factors_multiplier scale_factors_multiplier_inst (
    .sf_w_fp8(w_sf),
    .sf_a_fp8(a_sf),
    .is_int4_w(is_int4_w),
    .is_int4_a(is_int4_a),
    .sf_fp32(sf_fp32_comb)
);

genvar i;
generate
    for (i = 0; i < BLOCK_SIZE; i++) begin : g_mul
        w_a_multiplier w_a_multiplier_inst (
            .w_if4(w_in[i]),
            .a_if4(a_in[i]),
            .is_int4_w(is_int4_w),
            .is_int4_a(is_int4_a),
            .wa_fp16(wa_fp16_comb[i])
        );

        wa_sf_multiplier wa_sf_multiplier_inst (
            .wa_fp16(wa_fp16_comb[i]),
            .sf_fp32(sf_fp32_comb),
            .mul_fp32(mul_comb[i])
        );
    end

    // Stage 2 uses one FP32 adder per lane to keep a running accumulation.
    for (i = 0; i < BLOCK_SIZE; i++) begin : g_acc_lane
        CW_fp_add #(
            MANTISSA_BW,
            EXPONENT_BW,
            IEEE_COMPLIANCE,
            IEEE_NAN_COMPLIANCE,
            ARCH
        ) acc_lane_add_inst (
            .a(acc_lane[i]),
            .b(mul_stage1[i]),
            .rnd(3'b000),
            .z(acc_lane_add_out[i]),
            .status(acc_lane_add_status[i])
        );
    end

    // The final reduction reuses a tree of FP32 adders across cycles.
    for (i = 0; i < NUM_FP_ADD; i++) begin : g_reduce_add
        CW_fp_add #(
            MANTISSA_BW,
            EXPONENT_BW,
            IEEE_COMPLIANCE,
            IEEE_NAN_COMPLIANCE,
            ARCH
        ) reduce_add_inst (
            .a(reduce_vals[2*i]),
            .b(reduce_vals[2*i+1]),
            .rnd(3'b000),
            .z(reduce_add_out[i]),
            .status(reduce_add_status[i])
        );
    end
endgenerate

always_comb begin
    pair_count = reduce_count >> 1;
    next_count = pair_count + reduce_count[0];
end

// ------------------------
// Stage 1: MULTIPLICATION
// ------------------------
// Form wa_fp16, broadcast sf_fp32, and multiply to get mul_fp32 per lane.
// Capture the combinational multiply results and the corresponding first bit.
always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        stage1_valid <= 1'b0;
        first_stage1 <= 1'b0;
        for (int j = 0; j < BLOCK_SIZE; j++) begin
            mul_stage1[j] <= 32'd0;
        end
    end else begin
        stage1_valid <= en;
        if (en) begin
            first_stage1 <= first;
            for (int j = 0; j < BLOCK_SIZE; j++) begin
                mul_stage1[j] <= mul_comb[j];
            end
        end else begin
            first_stage1 <= 1'b0;
            for (int j = 0; j < BLOCK_SIZE; j++) begin
                mul_stage1[j] <= mul_stage1[j];
            end
        end
    end
end

// ------------------------
// Stage 2: ACCUMULATION
// ------------------------
// Each lane adds its product into a running accumulation for that lane. When
// the next first arrives, the completed batch is snapshotted for reduction while the new batch starts accumulating.
always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        batch_active <= 1'b0;
        snapshot_valid <= 1'b0;

        for (int j = 0; j < BLOCK_SIZE; j++) begin
            acc_lane[j] <= 32'd0;
            snapshot_vals[j] <= 32'd0;
        end
    end else begin
        if (stage1_valid) begin
            batch_active <= 1'b1;

            if (first_stage1 && batch_active) begin
                snapshot_valid <= 1'b1;
                for (int j = 0; j < BLOCK_SIZE; j++) begin
                    snapshot_vals[j] <= acc_lane[j];
                end
            end else begin
                snapshot_valid <= 1'b0;
                for (int j = 0; j < BLOCK_SIZE; j++) begin
                    snapshot_vals[j] <= snapshot_vals[j];
                end
            end

            for (int j = 0; j < BLOCK_SIZE; j++) begin
                acc_lane[j] <= first_stage1 ? mul_stage1[j] : acc_lane_add_out[j];
            end
        end else begin
            batch_active <= batch_active;
            snapshot_valid <= 1'b0;
            for (int j = 0; j < BLOCK_SIZE; j++) begin
                acc_lane[j] <= acc_lane[j];
                snapshot_vals[j] <= snapshot_vals[j];
            end
        end
    end
end

// ------------------------
// Stage 3+: FINAL REDUCTION
// ------------------------
// Reduce the snapshotted lane accumulators with a tree of FP32 adders until a single FP32 result remains.
always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        reduce_state <= REDUCE_IDLE;
        reduce_count <= '0;
        for (int j = 0; j < BLOCK_SIZE; j++) begin
            reduce_vals[j] <= 32'd0;
        end
    end else begin
        case (reduce_state)
            REDUCE_IDLE: begin
                if (snapshot_valid) begin
                    reduce_count <= REDUCE_CNT_W'(BLOCK_SIZE);
                    reduce_state <= REDUCE_STEP;
                    for (int j = 0; j < BLOCK_SIZE; j++) begin
                        reduce_vals[j] <= snapshot_vals[j];
                    end
                end else begin
                    reduce_state <= REDUCE_IDLE;
                    reduce_count <= reduce_count;
                    for (int j = 0; j < BLOCK_SIZE; j++) begin
                        reduce_vals[j] <= reduce_vals[j];
                    end
                end
            end

            REDUCE_STEP: begin
                if (next_count == REDUCE_CNT_W'(1)) begin
                    reduce_state <= REDUCE_IDLE;
                    reduce_count <= '0;
                end else begin
                    reduce_state <= REDUCE_STEP;
                    reduce_count <= next_count;
                end

                for (int j = 0; j < NUM_FP_ADD; j++) begin
                    if (j < pair_count) begin
                        reduce_vals[j] <= reduce_add_out[j];
                    end else begin
                        // This adder output is unused at the current reduction level.
                        reduce_vals[j] <= 32'd0;
                    end
                end

                if (reduce_count[0]) begin
                    reduce_vals[pair_count] <= reduce_vals[reduce_count-1];
                end

                for (int j = NUM_FP_ADD; j < BLOCK_SIZE; j++) begin
                    if (!(reduce_count[0] && (j == pair_count))) begin
                        // These entries are outside the live reduction frontier.
                        reduce_vals[j] <= 32'd0;
                    end
                end
            end

            default: begin
                reduce_state <= REDUCE_IDLE;
                reduce_count <= '0;
                for (int j = 0; j < BLOCK_SIZE; j++) begin
                    reduce_vals[j] <= 32'd0;
                end
            end
        endcase
    end
end

// Output the final accumulated result when reduction is done.
always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        acc <= 32'd0;
        done <= 1'b0;
    end
    else if (reduce_state == REDUCE_STEP && next_count == REDUCE_CNT_W'(1)) begin
        if (reduce_count == REDUCE_CNT_W'(1)) begin
            acc <= reduce_vals[0];
        end else begin
            acc <= reduce_add_out[0];
        end
        done <= 1'b1;
    end else begin
        acc <= acc;
        done <= 1'b0;
    end
end

endmodule // mac
