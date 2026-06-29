% Example: batched LP solves sharing one sparse matrix A.
% Each column of C, AL, AU, l, and u defines one LP instance.

fprintf('======================================================================\n');
fprintf('HPRLP Example: Batched Shared-A LP - MATLAB\n');
fprintf('======================================================================\n');

A = sparse([1.0, 2.0; 3.0, 1.0]);
B = 3;

C = [-3.0, -2.0, -4.0;
     -5.0, -6.0, -4.0];
AL = -inf(2, B);
AU = [10.0, 9.0, 11.0;
      12.0, 13.0, 11.0];
l = zeros(2, B);
u = inf(2, B);
u(1, 3) = 4.0;

params = hprlp.Parameters();
params.stop_tol = 1e-8;
params.max_iter = 200000;
params.use_presolve = false;

result = hprlp.solve_batched(A, C, AL, AU, l, u, 'params', params);

fprintf('Batch size: %d\n', numel(result.status));
fprintf('Total time: %.4f seconds\n', result.time);
for k = 1:numel(result.status)
    fprintf('[%d] status=%s obj=%.12e residual=%.6e iter=%d x=[%.6f %.6f]\n', ...
        k, result.status{k}, result.primal_obj(k), result.residuals(k), result.iter(k), result.x(1, k), result.x(2, k));
end
