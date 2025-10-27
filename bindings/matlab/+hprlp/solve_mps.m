function result = solve_mps(filename, param)
% solve_mps - Solve a linear programming problem from MPS file
%
% [DEPRECATED] This function is deprecated. Use the Model-based API instead:
%   model = hprlp.Model.from_mps(filename);
%   result = model.solve(param);
%
% Reads an LP problem from an MPS format file and solves it using HPRLP.
%
% Args:
%   filename - Path to MPS file (string)
%   param    - (Optional) Parameters object with solver settings
%
% Returns:
%   result - Result object containing solution, status, and timing
%
% Example:
%   % Solve with default parameters
%   result = hprlp.solve_mps('problem.mps');
%   
%   % Solve with custom parameters
%   param = hprlp.Parameters();
%   param.max_iter = 50000;
%   param.stop_tol = 1e-9;
%   param.verbose = true;
%   
%   result = hprlp.solve_mps('../../data/model.mps', param);
%   
%   if result.isOptimal()
%       fprintf('Optimal objective: %.6f\n', result.primal_obj);
%   end
%
% See also: hprlp.Model, hprlp.solve, hprlp.Parameters, hprlp.Result

    % Issue deprecation warning
    warning('HPRLP:Deprecated', ...
            ['hprlp.solve_mps() is deprecated. Use the Model-based API instead:\n' ...
             '  model = hprlp.Model.from_mps(filename);\n' ...
             '  result = model.solve(param);']);

    % Check inputs
    if nargin < 1
        error('HPRLP:InvalidInput', 'solve_mps requires filename argument');
    end
    
    % Create model from MPS file
    model = hprlp.Model.from_mps(filename);
    
    % Solve with or without parameters
    if nargin >= 2 && ~isempty(param)
        result = model.solve(param);
    else
        result = model.solve();
    end
    
    % Note: Model is automatically freed when it goes out of scope
end
