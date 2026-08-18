set script_dir [file dirname [file normalize [info script]]]
set project_root [file normalize [file join $script_dir ../..]]
set build_dir [expr {$argc > 0 ? [file normalize [lindex $argv 0]] :
    [file join $project_root build kv260_ooc]}]
file mkdir $build_dir
cd $project_root
create_project -force diffusion_accel_kv260 $build_dir -part xck26-sfvc784-2LV-c
set rtl_sources {}
foreach source [lsort [glob [file join $project_root rtl tensor_engine *.sv]]] {
    if {![string match "tb_*" [file tail $source]]} {
        lappend rtl_sources $source
    }
}
add_files -norecurse $rtl_sources
add_files -fileset constrs_1 -norecurse [file join $script_dir kernel_ooc.xdc]
add_files -norecurse [file join $project_root rtl tensor_engine exp_neg_q16_lut.hex]
set_property file_type {Memory Initialization Files} [get_files exp_neg_q16_lut.hex]
set_property top ddit_block_kv260_axi_top [current_fileset]
update_compile_order -fileset sources_1
synth_design -top ddit_block_kv260_axi_top -part xck26-sfvc784-2LV-c
report_utilization -file [file join $build_dir utilization.rpt]
report_timing_summary -delay_type max -max_paths 20 \
    -file [file join $build_dir timing_summary.rpt]
report_drc -file [file join $build_dir drc.rpt]
write_checkpoint -force [file join $build_dir ddit_block_kv260_axi_top_synth.dcp]
