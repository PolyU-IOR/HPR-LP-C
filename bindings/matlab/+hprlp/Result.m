classdef Result
    % Result - Solution results from HPRLP solver
    %
    % This class encapsulates the solution returned by the HPRLP solver,
    % including optimal values, solution vectors, iteration count, timing,
    % and convergence information.
    %
    % Properties:
    %   status      - Solution status: 'OPTIMAL', 'TIME_LIMIT', 'ITER_LIMIT', 'ERROR', etc.
    %   residuals   - Final KKT residual
    %   primal_obj  - Primal objective value c'*x
    %   gap         - Duality gap
    %   time4       - Time to reach 1e-4 tolerance (0 if not achieved)
    %   time6       - Time to reach 1e-6 tolerance (0 if not achieved)
    %   time8       - Time to reach 1e-8 tolerance (0 if not achieved)
    %   time        - Total solve time in seconds
    %   iter4       - Iterations to reach 1e-4 tolerance (0 if not achieved)
    %   iter6       - Iterations to reach 1e-6 tolerance (0 if not achieved)
    %   iter8       - Iterations to reach 1e-8 tolerance (0 if not achieved)
    %   iter        - Total number of iterations performed
    %   x           - Primal solution vector (n x 1)
    %   y           - Dual solution vector (m x 1)
    %
    % Example:
    %   result = hprlp.solve(A, AL, AU, l, u, c, param);
    %   if strcmp(result.status, 'OPTIMAL')
    %       fprintf('Optimal objective: %.6f\n', result.primal_obj);
    %       fprintf('Solution: x = [%.4f, %.4f]\n', result.x(1), result.x(2));
    %   end
    
    properties
        status      % Solution status string
        residuals   % Final KKT residual
        primal_obj  % Primal objective value
        gap         % Duality gap
        time4       % Time to reach 1e-4 tolerance
        time6       % Time to reach 1e-6 tolerance
        time8       % Time to reach 1e-8 tolerance
        time        % Total solve time (seconds)
        iter4       % Iterations to reach 1e-4 tolerance
        iter6       % Iterations to reach 1e-6 tolerance
        iter8       % Iterations to reach 1e-8 tolerance
        iter        % Total iteration count
        x           % Primal solution vector
        y           % Dual solution vector
    end
    
    methods
        function obj = Result(result_struct)
            % Result - Constructor from MEX result structure
            %
            % Args:
            %   result_struct - Structure returned by MEX interface
            
            if nargin > 0
                obj.status = result_struct.status;
                obj.residuals = result_struct.residuals;
                obj.primal_obj = result_struct.primal_obj;
                obj.gap = result_struct.gap;
                obj.time4 = result_struct.time4;
                obj.time6 = result_struct.time6;
                obj.time8 = result_struct.time8;
                obj.time = result_struct.time;
                obj.iter4 = result_struct.iter4;
                obj.iter6 = result_struct.iter6;
                obj.iter8 = result_struct.iter8;
                obj.iter = result_struct.iter;
                obj.x = result_struct.x;
                obj.y = result_struct.y;
            end
        end
        
        function disp(obj)
            % disp - Display solution results
            
            fprintf('HPRLP Solution Results:\n');
            fprintf('  Status:        %s\n', obj.status);
            fprintf('  Primal obj:    %.12e\n', obj.primal_obj);
            fprintf('  Gap:           %.6e\n', obj.gap);
            fprintf('  Residuals:     %.6e\n', obj.residuals);
            fprintf('  Iterations:    %d\n', obj.iter);
            fprintf('  Time:          %.4f s\n', obj.time);
            
            % Display milestone times/iterations if achieved
            if obj.time4 > 0
                fprintf('  Time to 1e-4:  %.4f s (iter: %d)\n', obj.time4, obj.iter4);
            end
            if obj.time6 > 0
                fprintf('  Time to 1e-6:  %.4f s (iter: %d)\n', obj.time6, obj.iter6);
            end
            if obj.time8 > 0
                fprintf('  Time to 1e-8:  %.4f s (iter: %d)\n', obj.time8, obj.iter8);
            end
            
            if ~isempty(obj.x)
                fprintf('  Solution dim:  n=%d, m=%d\n', length(obj.x), length(obj.y));
            end
        end
        
        function tf = isOptimal(obj)
            % isOptimal - Check if solution is optimal
            %
            % Returns:
            %   tf - True if status is 'OPTIMAL'
            
            tf = strcmp(obj.status, 'OPTIMAL');
        end
        
        function tf = isError(obj)
            % isError - Check if solver encountered an error
            %
            % Returns:
            %   tf - True if status is 'ERROR'
            
            tf = strcmp(obj.status, 'ERROR');
        end
    end
end
