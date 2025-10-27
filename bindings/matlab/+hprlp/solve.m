function result = solve(A, AL, AU, l, u, c, param)
% solve - Solve a linear programming problem with HPRLP
%
% [DEPRECATED] This function is deprecated. Use the Model-based API instead:
%   model = hprlp.Model.from_arrays(A, AL, AU, l, u, c);
%   result = model.solve(param);
%
% Solves an LP problem of the form:
%   minimize    c'*x
%   subject to  AL <= A*x <= AU
%               l <= x <= u
%
% Args:
%   A     - Constraint matrix (m x n sparse matrix)
%   AL    - Lower bounds for constraints (m x 1 vector, use -inf for unbounded)
%   AU    - Upper bounds for constraints (m x 1 vector, use inf for unbounded)
%   l     - Lower bounds for variables (n x 1 vector, use -inf for unbounded)
%   u     - Upper bounds for variables (n x 1 vector, use inf for unbounded)
%   c     - Objective coefficients (n x 1 vector)
%   param - (Optional) Parameters object with solver settings
%
% Returns:
%   result - Result object containing solution, status, and timing
%
% Example:
%   % Define problem: minimize -3*x1 - 5*x2
%   %                subject to x1 + 2*x2 <= 10
%   %                           3*x1 + x2 <= 12
%   %                           x1, x2 >= 0
%   A = sparse([1, 2; 3, 1]);
%   AL = [-inf; -inf];
%   AU = [10; 12];
%   l = [0; 0];
%   u = [inf; inf];
%   c = [-3; -5];
%   
%   param = hprlp.Parameters();
%   param.max_iter = 10000;
%   param.stop_tol = 1e-9;
%   
%   result = hprlp.solve(A, AL, AU, l, u, c, param);
%   
%   if result.isOptimal()
%       fprintf('Optimal solution: x = [%.4f, %.4f]\n', result.x(1), result.x(2));
%       fprintf('Objective value: %.6f\n', result.primal_obj);
%   end
%
% See also: hprlp.Model, hprlp.solve_mps, hprlp.Parameters, hprlp.Result

    % Issue deprecation warning
    warning('HPRLP:Deprecated', ...
            ['hprlp.solve() is deprecated. Use the Model-based API instead:\n' ...
             '  model = hprlp.Model.from_arrays(A, AL, AU, l, u, c);\n' ...
             '  result = model.solve(param);']);

    % Check inputs
    if nargin < 6
        error('HPRLP:InvalidInput', ...
              'solve requires at least 6 arguments: A, AL, AU, l, u, c');
    end
    
    % Create model from arrays
    model = hprlp.Model.from_arrays(A, AL, AU, l, u, c);
    
    % Solve with or without parameters
    if nargin >= 7 && ~isempty(param)
        result = model.solve(param);
    else
        result = model.solve();
    end
    
    % Note: Model is automatically freed when it goes out of scope
end
