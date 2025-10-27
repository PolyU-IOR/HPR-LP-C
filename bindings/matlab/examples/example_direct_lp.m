% Example: Solving a simple LP problem directly with HPRLP
%
% This example demonstrates how to solve a linear programming problem
% by directly providing the problem data using the Model-based API.
%
% Problem:
%     minimize    -3*x1 - 5*x2
%     subject to   x1 + 2*x2 <= 10
%                 3*x1 +  x2 <= 12
%                  x1, x2 >= 0
%
% Expected solution: x1 ≈ 2.8, x2 ≈ 3.6, objective ≈ -26.4

clear; clc;

% Add HPRLP to path
script_dir = fileparts(mfilename('fullpath'));
addpath(fullfile(script_dir, '..'));

fprintf('======================================================================\n');
fprintf('HPRLP Example: Direct LP Solution\n');
fprintf('======================================================================\n');
fprintf('\n');

%% Define the LP problem

% Constraint matrix in sparse format
% Row 1: x1 + 2*x2 <= 10
% Row 2: 3*x1 + x2 <= 12
A = sparse([1.0, 2.0;
            3.0, 1.0]);

% Constraint bounds (convert inequalities to double-sided bounds)
AL = [-inf; -inf];  % Lower bounds (no lower constraints)
AU = [10.0; 12.0];  % Upper bounds

% Variable bounds
l = [0.0; 0.0];     % Lower bounds (x >= 0)
u = [inf; inf];     % Upper bounds (unbounded above)

% Objective coefficients (minimize -3*x1 - 5*x2)
c = [-3.0; -5.0];

%% Create model

fprintf('Creating model...\n');
model = hprlp.Model.from_arrays(A, AL, AU, l, u, c);
fprintf('Model created: %d constraints, %d variables\n', model.m, model.n);
fprintf('\n');

%% Solve with parameters

fprintf('Solving with parameters (stop_tol=1e-9)...\n');
fprintf('----------------------------------------------------------------------\n');

param = hprlp.Parameters();
param.device_number = 0;
param.stop_tol = 1e-9;

result = model.solve(param);

%% Display results

fprintf('\n');
fprintf('======================================================================\n');
fprintf('Solution Results\n');
fprintf('======================================================================\n');
fprintf('\n');

fprintf('Status:         %s\n', result.status);
fprintf('Objective:      %.6f\n', result.primal_obj);
fprintf('Solution:       x = [%.6f, %.6f]\n', result.x(1), result.x(2));
fprintf('Iterations:     %d\n', result.iter);
fprintf('Time:           %.4f seconds\n', result.time);
fprintf('Duality gap:    %.6e\n', result.gap);
fprintf('Residuals:      %.6e\n', result.residuals);
fprintf('\n');

%% Verify solution

if strcmp(result.status, 'OPTIMAL')
    fprintf('Verification:\n');
    fprintf('  Constraint 1:  x1 + 2*x2 = %.6f <= 10\n', result.x(1) + 2*result.x(2));
    fprintf('  Constraint 2:  3*x1 + x2 = %.6f <= 12\n', 3*result.x(1) + result.x(2));
    fprintf('  Objective:     -3*x1 - 5*x2 = %.6f\n', -3*result.x(1) - 5*result.x(2));
    fprintf('\n');
    fprintf('✓ Solution is optimal!\n');
else
    fprintf('⚠ Solution is not optimal!\n');
end

fprintf('\n');
fprintf('======================================================================\n');

% Note: Model is automatically freed when it goes out of scope (destructor)