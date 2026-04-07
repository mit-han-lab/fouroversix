`timescale 1ns/1ps

module tb_nvfp4_mac;

localparam int BLOCK_SIZE = 16;
localparam int NUM_INPUTS = 31;
localparam int NUM_EXPECTED = 3;
localparam int DRAIN_CYCLES = 16;

logic        clk;
logic        rst_n;
logic        en;
logic        first;
logic [3:0]  a_in [BLOCK_SIZE-1:0];
logic [7:0]  a_sf;
logic [3:0]  w_in [BLOCK_SIZE-1:0];
logic [7:0]  w_sf;
logic [31:0] acc;
logic        done;

logic [3:0]  a_vec [NUM_INPUTS-1:0][BLOCK_SIZE-1:0];
logic [3:0]  w_vec [NUM_INPUTS-1:0][BLOCK_SIZE-1:0];
logic [7:0]  a_sf_vec [NUM_INPUTS-1:0];
logic [7:0]  w_sf_vec [NUM_INPUTS-1:0];
logic        first_vec [NUM_INPUTS-1:0];
logic [31:0] expected_acc [NUM_EXPECTED-1:0];

int expected_idx;

nvfp4_mac #(
    .BLOCK_SIZE(BLOCK_SIZE)
) dut (
    .clk(clk),
    .rst_n(rst_n),
    .en(en),
    .first(first),
    .a_in(a_in),
    .a_sf(a_sf),
    .w_in(w_in),
    .w_sf(w_sf),
    .acc(acc),
    .done(done)
);

always #5 clk = ~clk;

initial begin
    // Initialize all inputs to zero before deasserting reset.
    clk = 1'b0;
    rst_n = 1'b0;
    en = 1'b0;
    first = 1'b0;
    a_sf = '0;
    w_sf = '0;
    expected_idx = 0;

    for (int test_idx = 0; test_idx < NUM_INPUTS; test_idx++) begin
        first_vec[test_idx] = 1'b0;
        a_sf_vec[test_idx] = 8'd0;
        w_sf_vec[test_idx] = 8'd0;
        for (int lane = 0; lane < BLOCK_SIZE; lane++) begin
            a_vec[test_idx][lane] = 4'd0;
            w_vec[test_idx][lane] = 4'd0;
        end
    end

    for (int idx = 0; idx < NUM_EXPECTED; idx++) begin
        expected_acc[idx] = 32'd0;
    end

    // Load the precomputed stimulus stream and expected final accumulations.
`include "nvfp4_mac_golden_ref.svh"

    repeat (3) @(negedge clk);
    rst_n = 1'b1;

    // Drive one golden-reference input beat per cycle.
    for (int test_idx = 0; test_idx < NUM_INPUTS; test_idx++) begin
        @(negedge clk);
        en = 1'b1;
        first = first_vec[test_idx];
        a_sf = a_sf_vec[test_idx];
        w_sf = w_sf_vec[test_idx];
        for (int lane = 0; lane < BLOCK_SIZE; lane++) begin
            a_in[lane] = a_vec[test_idx][lane];
            w_in[lane] = w_vec[test_idx][lane];
        end
    end

    @(negedge clk);
    en = 1'b0;
    first = 1'b0;

    // Keep clocking so the final reduction can produce all expected outputs.
    repeat (DRAIN_CYCLES - 1) @(negedge clk);

    if (expected_idx != NUM_EXPECTED) begin
        $fatal(1, "Expected %0d done pulses but saw %0d", NUM_EXPECTED, expected_idx);
    end

    $display("PASS: tb_nvfp4_mac");
    $finish;
end

always @(posedge clk) begin
    if (rst_n && done) begin
        if (expected_idx >= NUM_EXPECTED) begin
            $fatal(1, "Unexpected extra done pulse with acc=%08h", acc);
        end
        // Each done pulse must match the next golden-reference final result.
        if (acc !== expected_acc[expected_idx]) begin
            $fatal(
                1,
                "Mismatch on result %0d: got %08h expected %08h",
                expected_idx,
                acc,
                expected_acc[expected_idx]
            );
        end
        $display("PASS: result[%0d] = %08h", expected_idx, acc);
        expected_idx = expected_idx + 1;
    end
end

endmodule
