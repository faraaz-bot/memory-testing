format longg

norms = [];

epsilon = 9.77e-04;

% Upper bound for diff.l_2 / cpu_output_norm.l_2 <= epsilon * sqrt(log2(total_length))
testdatahalf(:,6) = epsilon * sqrt(log2(testdatahalf(:,1)));

% Upper bound for diff.l_inf <= epsilon * cpu_output_norm.l_inf * log(total_length);
testdatahalf(:,7) = epsilon * testdatahalf(:,5) .* log2(testdatahalf(:,1));

tiledlayout(2,1)
nexttile
plot(testdatahalf(:,1), testdatahalf(:,2) ./ testdatahalf(:,3), 'LineWidth', 1, 'DisplayName', '|diff|_2/ |cpu|_2', ...
    'LineWidth', 1.5, 'color', 'blue')
hold on
plot(testdatahalf(:,1), testdatahalf(:,6), 'DisplayName', 'L_2 Cutoff')

title(['\fontsize{16} Error Analysis for L_2 norm'])
legend()
xlabel('FFT size')

nexttile
plot(testdatahalf(:,1), testdatahalf(:,4), 'LineWidth', 1,  'DisplayName', '|diff|_{inf}', ...
    'LineWidth', 1.5, 'color', 'green')
 hold on
plot(testdatahalf(:,1), testdatahalf(:,7), 'DisplayName', 'L_{inf} Cutoff')

title(['\fontsize{16} Error Analysis for L_{inf} norm'])
legend()
xlabel('FFT size')
