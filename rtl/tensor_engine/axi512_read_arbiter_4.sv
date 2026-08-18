`timescale 1ns/1ps

module axi512_read_arbiter_4 (
    input  wire clk,
    input  wire rst_n,
    input  wire [4*64-1:0] s_axi_araddr,
    input  wire [4*8-1:0] s_axi_arlen,
    input  wire [4*3-1:0] s_axi_arsize,
    input  wire [4*2-1:0] s_axi_arburst,
    input  wire [3:0] s_axi_arvalid,
    output reg  [3:0] s_axi_arready,
    output wire [511:0] s_axi_rdata,
    output wire [1:0] s_axi_rresp,
    output wire s_axi_rlast,
    output reg  [3:0] s_axi_rvalid,
    input  wire [3:0] s_axi_rready,
    output wire [63:0] m_axi_araddr,
    output wire [7:0] m_axi_arlen,
    output wire [2:0] m_axi_arsize,
    output wire [1:0] m_axi_arburst,
    output wire m_axi_arvalid,
    input  wire m_axi_arready,
    input  wire [511:0] m_axi_rdata,
    input  wire [1:0] m_axi_rresp,
    input  wire m_axi_rlast,
    input  wire m_axi_rvalid,
    output wire m_axi_rready,
    output wire busy,
    output reg  protocol_error,
    output reg  [63:0] read_transactions,
    output reg  [63:0] arbitration_cycles
);

    localparam [1:0] IDLE=0,ADDRESS=1,DATA=2;
    reg [1:0] state;
    reg [1:0] active_grant;
    reg [1:0] last_grant;
    reg [8:0] active_beats;
    reg [8:0] accepted_beats;
    reg [1:0] selected_grant;
    reg selected_valid;
    wire accepting_data=m_axi_rvalid && m_axi_rready;
    wire expected_last=accepted_beats==active_beats-1'b1;

    always @* begin
        selected_valid=1'b1;selected_grant=0;
        case(last_grant)
            2'd0: begin
                if(s_axi_arvalid[1]) selected_grant=1;
                else if(s_axi_arvalid[2]) selected_grant=2;
                else if(s_axi_arvalid[3]) selected_grant=3;
                else if(s_axi_arvalid[0]) selected_grant=0;
                else selected_valid=0;
            end
            2'd1: begin
                if(s_axi_arvalid[2]) selected_grant=2;
                else if(s_axi_arvalid[3]) selected_grant=3;
                else if(s_axi_arvalid[0]) selected_grant=0;
                else if(s_axi_arvalid[1]) selected_grant=1;
                else selected_valid=0;
            end
            2'd2: begin
                if(s_axi_arvalid[3]) selected_grant=3;
                else if(s_axi_arvalid[0]) selected_grant=0;
                else if(s_axi_arvalid[1]) selected_grant=1;
                else if(s_axi_arvalid[2]) selected_grant=2;
                else selected_valid=0;
            end
            default: begin
                if(s_axi_arvalid[0]) selected_grant=0;
                else if(s_axi_arvalid[1]) selected_grant=1;
                else if(s_axi_arvalid[2]) selected_grant=2;
                else if(s_axi_arvalid[3]) selected_grant=3;
                else selected_valid=0;
            end
        endcase
    end

    assign m_axi_araddr=s_axi_araddr[active_grant*64 +: 64];
    assign m_axi_arlen=s_axi_arlen[active_grant*8 +: 8];
    assign m_axi_arsize=s_axi_arsize[active_grant*3 +: 3];
    assign m_axi_arburst=s_axi_arburst[active_grant*2 +: 2];
    assign m_axi_arvalid=state==ADDRESS;
    assign s_axi_rdata=m_axi_rdata;
    assign s_axi_rresp=m_axi_rresp;
    assign s_axi_rlast=m_axi_rlast;
    assign m_axi_rready=state==DATA && s_axi_rready[active_grant];
    assign busy=state!=IDLE;

    always @* begin
        s_axi_arready=0;s_axi_rvalid=0;
        if(state==ADDRESS) s_axi_arready[active_grant]=m_axi_arready;
        if(state==DATA) s_axi_rvalid[active_grant]=m_axi_rvalid;
    end

    always @(posedge clk) begin
        if(!rst_n) begin
            state<=IDLE;active_grant<=0;last_grant<=3;
            active_beats<=0;accepted_beats<=0;protocol_error<=0;
            read_transactions<=0;arbitration_cycles<=0;
        end else begin
            if(state==IDLE) begin
                if(selected_valid) begin
                    active_grant<=selected_grant;state<=ADDRESS;
                    arbitration_cycles<=arbitration_cycles+1'b1;
                end
            end else if(state==ADDRESS && m_axi_arready) begin
                active_beats<={1'b0,m_axi_arlen}+1'b1;
                accepted_beats<=0;read_transactions<=read_transactions+1'b1;
                state<=DATA;
            end else if(state==DATA && accepting_data) begin
                if(m_axi_rresp!=0 || m_axi_rlast!=expected_last)
                    protocol_error<=1'b1;
                if(m_axi_rlast || expected_last) begin
                    last_grant<=active_grant;state<=IDLE;
                end else accepted_beats<=accepted_beats+1'b1;
            end
        end
    end
endmodule
