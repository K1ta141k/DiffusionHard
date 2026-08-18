set script_dir [file dirname [file normalize [info script]]]
set project_dir [file join $script_dir build candidate_sampler_hls]

if {![info exists ::env(DIFFUSION_ACCEL_FPGA_PART)]} {
    error "Set DIFFUSION_ACCEL_FPGA_PART to the installed Vitis HLS part name"
}

set clock_period_ns 3.333
if {[info exists ::env(DIFFUSION_ACCEL_CLOCK_PERIOD_NS)]} {
    set clock_period_ns $::env(DIFFUSION_ACCEL_CLOCK_PERIOD_NS)
}

open_project -reset $project_dir
set_top candidate_reveal_kernel
add_files [file join $script_dir candidate_sampler.cpp]
add_files [file join $script_dir candidate_sampler.hpp]
add_files -tb [file join $script_dir candidate_sampler_test.cpp]

open_solution -reset solution1
set_part $::env(DIFFUSION_ACCEL_FPGA_PART)
create_clock -period $clock_period_ns -name default

csim_design
csynth_design
export_design -format ip_catalog
exit
