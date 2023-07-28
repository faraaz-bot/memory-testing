function [kernel_3d, kernel_3d_partial] = build_kernels_3d(kernel, item_sz_bytes, ...
                                                            max_lds_sz_bytes, max_ram_sz_bytes)
  
  num_kernels = length(kernel)
  kernel_count = 1;
  kernel_partial_pass_count = 1;
  
  tpb_1d_partial = 1;
  tpb_2d = 1;
  for i=1:num_kernels
    kernel(i).tpb = kernel(i).wgs / kernel(i).tpt;
    kernel(i).prime_factors = factor(kernel(i).length)';
    lengths(i) = kernel(i).length;
  endfor
  
  # Computes all k-combinations (k=3) with repetitions of 
  # the supported 1-D lengths. 
  # Ex: [32, 32, 64] is a valid combinations
  # The combinations [16, 32, 64] and [32, 16, 64] are considered
  # to be the same.
  # Equation for total number of combinations is:
  # num_3d_kernels = factorial(n+k-1) / (factorial(k) * factorial(n-1))
  # where n = num_kernels and k = 3;
  lengths_3d = nmultichoosek(lengths,3);    
  num_3d_kernels = size(lengths_3d, 1);  
     
  kernel_3d = struct('index', repmat({[]}, 1, num_3d_kernels));
  kernel_3d_partial = struct('index', repmat({[]}, 1, num_3d_kernels));
  for i=1:num_3d_kernels
    length1 = lengths_3d(i,1);
    length2 = lengths_3d(i,2);
    length3 = lengths_3d(i,3);
    
    idx_kernel1 = find(lengths == length1);
    idx_kernel2 = find(lengths == length2);
    idx_kernel3 = find(lengths == length3);
    
    factors1 = kernel(idx_kernel1).factors;
    factors2 = kernel(idx_kernel2).factors;
    factors3 = kernel(idx_kernel3).factors;
    
    prime_factors1 = kernel(idx_kernel1).prime_factors;
    prime_factors2 = kernel(idx_kernel2).prime_factors;
    prime_factors3 = kernel(idx_kernel3).prime_factors;
        
    length_fits_in_ram = (length1*length2*length3*item_sz_bytes) <= max_ram_sz_bytes;
       
    cond_lds_1_2d = (length1*length2*tpb_2d*item_sz_bytes) <= max_lds_sz_bytes;
    cond_lds_2_2d = (length1*length3*tpb_2d*item_sz_bytes) <= max_lds_sz_bytes;
    cond_lds_3_2d = (length2*length3*tpb_2d*item_sz_bytes) <= max_lds_sz_bytes;
    
    length_fits_in_lds = (cond_lds_1_2d || cond_lds_2_2d || cond_lds_3_2d);
    
    if (length_fits_in_ram && length_fits_in_lds)
      % Three kernels can be trivially reduced to 2 kernels that fit into LDS
      kernel_3d(kernel_count).prev_length = [length1;length2;length3];
      kernel_3d(kernel_count).tpt = tpb_2d;      
      if cond_lds_1_2d
        kernel_3d(kernel_count).length = [length1*length2;length3];
        kernel_3d(kernel_count).factors_1 = [factors1;factors2];
        kernel_3d(kernel_count).factors_2 = factors3;
      elseif cond_lds_2_2d
        kernel_3d(kernel_count).length = [length1*length3;length2];
        kernel_3d(kernel_count).factors_1 = [factors1;factors3];
        kernel_3d(kernel_count).factors_2 = factors2;
      elseif cond_lds_3_2d
        kernel_3d(kernel_count).length = [length2*length3;length1];
        kernel_3d(kernel_count).factors_1 = [factors2;factors3];
        kernel_3d(kernel_count).factors_2 = factors1;
      endif
     
      kernel_count = kernel_count + 1;
    elseif (length_fits_in_ram)                        
      # Try partial pass in first off dimension 
      if ~((length2*tpb_1d_partial*item_sz_bytes>=max_lds_sz_bytes) && ...
           (length3*tpb_1d_partial*item_sz_bytes>=max_lds_sz_bytes))
        [new_factors2, new_factors3] = solve_prod_partition_problem(prime_factors1, factors2, factors3);      
        cond_lds_2d = ((prod(new_factors2)*tpb_1d_partial*item_sz_bytes) <= max_lds_sz_bytes) && ...
                      ((prod(new_factors3)*tpb_1d_partial*item_sz_bytes) <= max_lds_sz_bytes);
        if (cond_lds_2d)
          kernel_3d_partial(kernel_partial_pass_count).length = [prod(new_factors2);prod(new_factors3)];
          kernel_3d_partial(kernel_partial_pass_count).tpt = tpb_1d_partial;
          kernel_3d_partial(kernel_partial_pass_count).factors_1 = new_factors2;
          kernel_3d_partial(kernel_partial_pass_count).factors_2 = new_factors3;
          kernel_3d_partial(kernel_partial_pass_count).prev_length = [length1;length2;length3];
        
          kernel_partial_pass_count = kernel_partial_pass_count + 1;        
          continue;
        endif
      endif      
      
      # Try partial pass in second off dimension
      if ~((length1*tpb_1d_partial*item_sz_bytes>=max_lds_sz_bytes) && ...
           (length3*tpb_1d_partial*item_sz_bytes>=max_lds_sz_bytes))           
        [new_factors1, new_factors3] = solve_prod_partition_problem(prime_factors2, factors1, factors3);
        cond_lds_2d = ((prod(new_factors1)*tpb_1d_partial*item_sz_bytes) <= max_lds_sz_bytes) && ...
                      ((prod(new_factors3)*tpb_1d_partial*item_sz_bytes) <= max_lds_sz_bytes);
        if (cond_lds_2d)
          kernel_3d_partial(kernel_partial_pass_count).length = [prod(new_factors1);prod(new_factors3)];
          kernel_3d_partial(kernel_partial_pass_count).tpt = tpb_1d_partial;
          kernel_3d_partial(kernel_partial_pass_count).factors_1 = new_factors1;
          kernel_3d_partial(kernel_partial_pass_count).factors_2 = new_factors3;
          kernel_3d_partial(kernel_partial_pass_count).prev_length = [length1;length2;length3];
        
          kernel_partial_pass_count = kernel_partial_pass_count + 1;
          continue;
        endif           
      endif

      # Try partial pass in third off dimension
      if ~((length1*tpb_1d_partial*item_sz_bytes>=max_lds_sz_bytes) && ...
           (length2*tpb_1d_partial*item_sz_bytes>=max_lds_sz_bytes))
        [new_factors1, new_factors2] = solve_prod_partition_problem(prime_factors3, factors1, factors2);
        cond_lds_2d = ((prod(new_factors1)*tpb_1d_partial*item_sz_bytes) <= max_lds_sz_bytes) && ...
                      ((prod(new_factors2)*tpb_1d_partial*item_sz_bytes) <= max_lds_sz_bytes);
        if (cond_lds_2d)
          kernel_3d_partial(kernel_partial_pass_count).length = [prod(new_factors1);prod(new_factors2)];
          kernel_3d_partial(kernel_partial_pass_count).tpt = tpb_1d_partial;
          kernel_3d_partial(kernel_partial_pass_count).factors_1 = new_factors1;
          kernel_3d_partial(kernel_partial_pass_count).factors_2 = new_factors2;
          kernel_3d_partial(kernel_partial_pass_count).prev_length = [length1;length2;length3];
        
          kernel_partial_pass_count = kernel_partial_pass_count + 1;
          continue;
        endif           
      endif      
      
    endif  
    
    printf('%d of %d \n', i, num_3d_kernels);
  endfor  
  
  kernel_3d = kernel_3d(1:kernel_count-1);
  kernel_3d_partial = kernel_3d_partial(1:kernel_partial_pass_count-1);

endfunction