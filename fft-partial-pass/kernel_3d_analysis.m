function kernel_3d_analysis()

precision_single_bytes=8;
precision_double_bytes=16;
max_ram_sz_bytes = 128 * 10^9; 
max_lds_sz_bytes = 32 * 10^3;

kernel = load_kernels();

precision = precision_single_bytes;
[kernel_3d, kernel_3d_partial] = build_kernels_3d(kernel, precision, max_lds_sz_bytes, max_ram_sz_bytes);
save_kernels('length_cs_kernel_2d_single.dat', kernel_3d);
save_kernels('length_partial_pass_single.dat', kernel_3d_partial);


precision = precision_double_bytes;
[kernel_3d, kernel_3d_partial] = build_kernels_3d(kernel, precision, max_lds_sz_bytes, max_ram_sz_bytes);
save_kernels('length_cs_kernel_2d_double.dat', kernel_3d);
save_kernels('length_partial_pass_double.dat', kernel_3d_partial);

endfunction
