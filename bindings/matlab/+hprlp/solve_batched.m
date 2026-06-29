function result = solve_batched(A, C, AL, AU, l, u, varargin)
% solve_batched - Solve LPs that share one matrix A.
%
% result = hprlp.solve_batched(A, C, AL, AU, l, u)
% result = hprlp.solve_batched(..., 'obj_constants', obj, 'params', params)
%
% C, l, and u are n-by-B. AL and AU are m-by-B. Each column is one LP.

    p = inputParser;
    addParameter(p, 'obj_constants', [], @(x) isempty(x) || isnumeric(x));
    addParameter(p, 'params', [], @(x) isempty(x) || isa(x, 'hprlp.Parameters'));
    parse(p, varargin{:});

    [m, n] = size(A);
    model = hprlp.Model.from_arrays(A, zeros(m, 1), zeros(m, 1), zeros(n, 1), zeros(n, 1), zeros(n, 1));
    cleanup = onCleanup(@() delete(model));
    result = model.solve_batched(C, AL, AU, l, u, ...
        'obj_constants', p.Results.obj_constants, ...
        'params', p.Results.params);
end
