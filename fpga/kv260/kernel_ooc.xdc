create_clock -name ap_clk -period 4.000 [get_ports ap_clk]
set_clock_uncertainty 0.200 [get_clocks ap_clk]
