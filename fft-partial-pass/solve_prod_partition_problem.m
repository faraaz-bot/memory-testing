%%%%%%%%%%%%%%%%%%%
% Usage example
%%%%%%%%%%%%%%%%%%%
% n = 12;
% a = 1;
% b = 17;  
% input_set = round(a + (b-a) * rand(n,1)); 
% fixed_set1 = [10; 5; 3];
% fixed_set2 = [9; 1; 8; 15];
% [output_set1, output_set2] = solve_prod_partition_problem(input_set, fixed_set1, fixed_set2);

function [output_set1, output_set2] = solve_prod_partition_problem(input_set, fixed_set1, fixed_set2)
  
  n = length(input_set);
  prod_total = prod(input_set);    
  
  offset_prod1 = prod(fixed_set1);  
  offset_prod2 = prod(fixed_set2);
  
  [output_set1,output_set2, prod_diff] = find_min_recursive(input_set, n + 1, 1, ...
                                          prod_total, [], [], offset_prod1, offset_prod2);
            
  output_set1 = [fixed_set1; output_set1];
  output_set2 = [fixed_set2; output_set2];
  
  function [set1, set2, prod_diff] = find_min_recursive(values, i, prod_calculated, ...
                                    prod_total, set1, set2, offset_prod1, offset_prod2)
    if (i == 1)              
        prod_set1 = prod_calculated * offset_prod1;
        
        set2 = setdiff_(values, set1);                
        prod_set2 = prod(set2) * offset_prod2;        
        prod_diff = abs(prod_set1-prod_set2); 
        
        return;
    endif
    
    
    [set1_1, set2_1, prod_diff_1] = find_min_recursive(values, i - 1, prod_calculated * values(i - 1), ...
                                       prod_total, [set1; values(i - 1)], set2, offset_prod1, offset_prod2);
    [set1_2, set2_2, prod_diff_2] = find_min_recursive(values, i - 1, prod_calculated, ...
                                       prod_total, set1, set2, offset_prod1, offset_prod2);
    
    if (prod_diff_1 <= prod_diff_2)     
      prod_diff = prod_diff_1;
      set1 = set1_1;
      set2 = set2_1;
    else     
      prod_diff = prod_diff_2;
      set1 = set1_2;
      set2 = set2_2;
    endif
     
    return;
  endfunction
  
  function c = setdiff_(a, b)  
    n = length(b);      
    a_copy = a;
      
    for i=1:n
      idx=find(a_copy==b(i));
      if (~isempty(idx))
        a_copy(idx(1)) = [];
      endif
    endfor
      
    c = a_copy;
  endfunction
    
endfunction
