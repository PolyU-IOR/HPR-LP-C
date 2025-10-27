% Example: Solving LP from MPS file with HPRLP
%
% This example demonstrates how to solve a linear programming problem
% by reading it from an MPS format file using the Model-based API.
% Shows model reuse with different parameters.
%
% The example uses the model.mps file from the data directory, which contains:
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
fprintf('HPRLP Example: Solving from MPS File\n');
fprintf('======================================================================\n');
fprintf('\n');

%% Locate MPS file

% The MPS file is in the data directory at the root of the project
script_dir = fileparts(mfilename('fullpath'));
mps_file = fullfile(script_dir, '..', '..', '..', 'data', 'model.mps');

% Check if file exists
if ~exist(mps_file, 'file')
    error('MPS file not found: %s\nPlease ensure the data/model.mps file exists.', mps_file);
end

fprintf('MPS file: %s\n', mps_file);
fprintf('\n');

%% Create model from MPS file

fprintf('Creating model from MPS file...\n');
model = hprlp.Model.from_mps(mps_file);
fprintf('Model created: %d constraints, %d variables\n', model.m, model.n);
fprintf('\n');

%% Solve with custom parameters

param1 = hprlp.Parameters();
param1.device_number = 0;
param1.stop_tol = 1e-9;

fprintf('Solve with custom parameters:\n');
fprintf('\n');

result1 = model.solve(param1);

%% Display results

fprintf('======================================================================\n');
fprintf('Solution Results\n');
fprintf('======================================================================\n');
fprintf('\n');

fprintf('Status:         %s\n', result1.status);
fprintf('Objective:      %.6f\n', result1.primal_obj);
fprintf('Iterations:     %d\n', result1.iter);
fprintf('Time:           %.4f seconds\n', result1.time);
fprintf('Duality gap:    %.6e\n', result1.gap);
fprintf('Residuals:      %.6e\n', result1.residuals);
fprintf('\n');
