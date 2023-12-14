function partial_pass_test()
  % Input parameters
  in_length = [64 64 64];  
  N = prod(in_length);
  
  pp_dim = 2;    
  pp_radices = [8 8];  
  pp_mode = 'six-step';
    
  % Generate input data  
  in = complex(0:N-1,0:N-1);
  % in = randn(in_length);
  in = reshape(in, in_length(1), in_length(2), in_length(3));
  
  % 3D-FFT
  out = fftn(in);  
  
  % CS_3D_RC
  [out_3d_rc, out_3d_rc_pp_1, out_3d_rc_pp_2] = run_CS_3D_RC(in_length, in, pp_dim, pp_radices, pp_mode);        
  
  % CS_TRTRTR
  % [out_trtrtr, out_trtrtr_pp_1, out_trtrtr_pp_2] = run_CS_TRTRTR(in_length, in, pp_dim, pp_radices);  
    
  % Validate output
  out_flat = reshape(out, 1, []);
  out_3d_rc_flat = reshape(out_3d_rc, 1, []);  
  % out_3d_rc_pp_1_flat = reshape(out_3d_rc_pp_1, 1, []);
  % out_3d_rc_pp_2_flat = reshape(out_3d_rc_pp_2, 1, []);
  % out_trtrtr_flat = reshape(out_trtrtr, 1, []);  
    
  disp(norm(out_flat-out_3d_rc_flat, 'inf'));  
  % disp(norm(out_flat-out_trtrtr_flat, 'inf'));   
  
  function [out, out_pp_1, out_pp_2] = run_CS_3D_RC(in_length, in, pp_dim, pp_radices, pp_mode)
    n = in_length(pp_dim);
    n1 = pp_radices(1);
    n2 = pp_radices(2);
    
    F_n = dft_matrix(n);  
    F_n1 = dft_matrix(n1);
    F_n2 = dft_matrix(n2);
    
    out = in;
  
    if (pp_dim == 1)
      % 1st kernel (2nd dimension)
      out = fft(out,[], 2);
      out = partial_pass_step_1_2(out, 1, n1, n2, F_n1, F_n2, F_n, pp_mode);
      out_pp_1 = out;
      
      % 2nd kernel (3rd dimension)
      out = partial_pass_step_3_4(out, 1, n1, n2, F_n1, F_n2, F_n, pp_mode);
      out_pp_2 = out;
      out = fft(out,[], 3);
    endif
  
    if (pp_dim == 2)
      % 1st kernel (1st dimension)
      out = fft(out,[], 1);       
      out = partial_pass_step_1_2(out, 2, n1, n2, F_n1, F_n2, F_n, pp_mode);    
      out_pp_1 = out;
            
      % 2nd kernel (3rd dimension)
      out = partial_pass_step_3_4(out, 2, n1, n2, F_n1, F_n2, F_n, pp_mode);
      out_pp_2 = out;
      out = fft(out,[], 3);
    endif
  
    if (pp_dim == 3)               
      % 1st kernel (1st dimension)
      out = fft(out,[], 1); 
      out = partial_pass_step_1_2(out, 3, n1, n2, F_n1, F_n2, F_n, pp_mode);    
      out_pp_1 = out;
      
      % 2nd kernel (2nd dimension)
      out = partial_pass_step_3_4(out, 3, n1, n2, F_n1, F_n2, F_n, pp_mode);
      out_pp_2 = out;
      out = fft(out,[], 2);
    endif
  endfunction
  
  function [out, out_pp_1, out_pp_2] = run_CS_TRTRTR(in_length, in, pp_dim, pp_radices)
    n = in_length(pp_dim);  
    n1 = pp_radices(1);
    n2 = pp_radices(2);
    
    F_n = dft_matrix(n);  
    F_n1 = dft_matrix(n1);
    F_n2 = dft_matrix(n2);
    
    transp_order = [2 3 1];    

    if (pp_dim == 1)         
      out = permute(in, transp_order); 
      out = permute(out, transp_order);
      out = fft(out,[], 1);  
      out = partial_pass_step_1_2(out, 3, n1, n2, F_n1, F_n2, F_n); %TODO: figure out input arg here 1,2,3?
      out_pp_1 = out;
      out = permute(out, transp_order);
      out = partial_pass_step_3_4(out, 3, n1, n2, F_n1, F_n2, F_n); %TODO: figure out input arg here 1,2,3?
      out_pp_2 = out;
      out = fft(out,[], 1);  
    endif  
    
    if (pp_dim == 2)            
      out = permute(in, transp_order); 
      out = fft(out,[], 1);    
      out = partial_pass_step_1_2(out, 1, n1, n2, F_n1, F_n2, F_n);    
      out_pp_1 = out;
      out = permute(out, transp_order);
      out = permute(out, transp_order);
      out = partial_pass_step_3_4(out, 1, n1, n2, F_n1, F_n2, F_n);
      out_pp_2 = out;
      out = fft(out,[], 1);  
    endif
    
    if (pp_dim == 3)            
      out = permute(in, transp_order); 
      out = fft(out,[], 1);    
      out = partial_pass_step_1_2(out, 1, n1, n2, F_n1, F_n2, F_n);    
      out_pp_1 = out;
      out = permute(out, transp_order);
      out = fft(out,[], 1);  
      out = partial_pass_step_3_4(out, 1, n1, n2, F_n1, F_n2, F_n);
      out_pp_2 = out;
      out = permute(out, transp_order);
    endif    
  endfunction
  
  function [dim1, dim2] = get_data_dim_partial_pass(input, pp_dim)
    if (pp_dim==1)
      dim1 = size(input,2);
      dim2 = size(input,3);
    elseif (pp_dim==2)
      dim1 = size(input,1);
      dim2 = size(input,3);
    elseif (pp_dim==3)
      dim1 = size(input,1);
      dim2 = size(input,2);
    endif
  endfunction
  
  function input_data_decomp = get_pp_decomposed_data(input_data, pp_dim, idx1, idx2, n1, n2)
    if (pp_dim==1)
      input_data_decomp = reshape(input_data(:,idx1,idx2), n1, n2);
    elseif (pp_dim==2)
      input_data_decomp = reshape(input_data(idx1,:,idx2), n1, n2);
    elseif (pp_dim==3)
      input_data_decomp = reshape(input_data(idx1,idx2,:), n1, n2);
    endif    
  endfunction
  
  function output = set_pp_data(input, input_decomp, pp_dim, idx1, idx2)
    output = input;
    
    if (pp_dim==1)
      output(:,idx1,idx2) = reshape(input_decomp, [], 1);
    elseif (pp_dim==2)
      output(idx1,:,idx2) = reshape(input_decomp, [], 1);
    elseif (pp_dim==3)
      output(idx1,idx2,:) = reshape(input_decomp, [], 1);
    endif    
  endfunction
  
  function output = partial_pass_step_1_2(input, pp_dim, n1, n2, F_n1, F_n2, F_n, mode)
    output = input;    
    
    [dim1, dim2] = get_data_dim_partial_pass(input, pp_dim);
        
    for idx1=1:dim1
      for idx2=1:dim2
        in_decomp = get_pp_decomposed_data(output, pp_dim, idx1, idx2, n1, n2);
        
        if strcmp(mode, 'four-step')
          % Length n2 FFT 
          out_decomp = in_decomp*F_n2;
          % Twiddle multiply
          out_decomp = F_n(1:n1, 1:n2).*out_decomp;
        elseif strcmp(mode, 'six-step')
          % Local transpose
          out_decomp = in_decomp.';          
          % Length n1 FFT 
          out_decomp = F_n1*out_decomp;
          % Twiddle multiply
          out_decomp = F_n(1:n1, 1:n2).*out_decomp;
        else
          error('invalid partial-pass mode');
        endif                     
      
        output = set_pp_data(output, out_decomp, pp_dim, idx1, idx2);
      endfor
    endfor
  endfunction
  
  function output = partial_pass_step_3_4(input, pp_dim, n1, n2, F_n1, F_n2, F_n, mode)
    output = input;
    
    [dim1, dim2] = get_data_dim_partial_pass(input, pp_dim);
    
    for idx1=1:dim1
      for idx2=1:dim2
        in_decomp = get_pp_decomposed_data(output, pp_dim, idx1, idx2, n1, n2);
        
        if strcmp(mode, 'four-step')
          % Local transpose
          out_decomp = in_decomp.';
          % Length n1 FFT
          out_decomp = out_decomp*F_n1;
        elseif strcmp(mode, 'six-step')
          % Local transpose
          out_decomp = in_decomp.';
          % Length n2 FFT
          out_decomp = F_n2*out_decomp;
          % Local transpose
          out_decomp = out_decomp.';
        else
          error('invalid partial-pass mode');
        endif                                  
      
        output = set_pp_data(output, out_decomp, pp_dim, idx1, idx2);
      endfor
    endfor
  endfunction
  
endfunction