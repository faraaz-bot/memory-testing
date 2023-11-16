%%%%%%%%%%%%%%%%%%%
% Usage example
%%%%%%%%%%%%%%%%%%%
% n = 12;
% a = 1;
% b = 17;  
% input_set = round(a + (b-a) * rand(n,1)); 
% fixed_set1 = [10; 5; 3];
% fixed_set2 = [9; 1; 8; 15];
% [output_set1, output_set2] = solve_sum_partition_problem(input_set, fixed_set1, fixed_set2);

function [output_set1, output_set2] = solve_sum_partition_problem(input_set, fixed_set1, fixed_set2)
  
  n = length(input_set);
  sum_total = sum(input_set);    
  
  offset_sum1 = sum(fixed_set1);  
  offset_sum2 = sum(fixed_set2);
  
  [output_set1,output_set2, sum_diff] = find_min_recursive(input_set, n + 1, 0, ...
                                          sum_total, [], [], offset_sum1, offset_sum2);
      
  output_set1 = [fixed_set1; output_set1];
  output_set2 = [fixed_set2; output_set2];
  
  function [set1, set2, sum_diff] = find_min_recursive(values, i, sum_calculated, ...
                                    sum_total, set1, set2, offset_sum1, offset_sum2)
    if (i == 1)              
        sum_set1 = sum_calculated + offset_sum1;
        
        set2 = setdiff_(values, set1);
        sum_set2 = (sum_total - sum_calculated) + offset_sum2;
        
        sum_diff = abs(sum_set1-sum_set2);               
        
        return;
    endif
    
    
    [set1_1, set2_1, sum_diff_1] = find_min_recursive(values, i - 1, sum_calculated + values(i - 1), ...
                                       sum_total, [set1; values(i - 1)], set2, offset_sum1, offset_sum2);
    [set1_2, set2_2, sum_diff_2] = find_min_recursive(values, i - 1, sum_calculated, ...
                                       sum_total, set1, set2, offset_sum1, offset_sum2);
    
    if (sum_diff_1 <= sum_diff_2)     
      sum_diff = sum_diff_1;
      set1 = set1_1;
      set2 = set2_1;
    else     
      sum_diff = sum_diff_2;
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
