classdef Parameters
    % Parameters - Configuration parameters for HPRLP solver
    %
    % This class holds all configuration parameters for the HPRLP solver,
    % including iteration limits, tolerances, GPU device selection.
    %
    % Properties:
    %   max_iter                  - Maximum number of iterations (default: INT32_MAX)
    %   stop_tol                  - Stopping tolerance for KKT residual (default: 1e-4)
    %   time_limit                - Time limit in seconds (default: 3600.0)
    %   device_number             - CUDA device ID to use (default: 0)
    %   check_iter                - Check convergence every N iterations (default: 150)
    %   use_Ruiz_scaling          - Use Ruiz scaling (default: true)
    %   use_Pock_Chambolle_scaling - Use Pock-Chambolle scaling (default: true)
    %   use_bc_scaling            - Use bound constraint scaling (default: true)
    %
    % Example:
    %   param = hprlp.Parameters();
    %   param.max_iter = 50000;
    %   param.stop_tol = 1e-6;
    %   param.device_number = 0;
    %   result = hprlp.solve(A, AL, AU, l, u, c, param);
    
    properties
        max_iter                  = 2147483647  % Maximum number of iterations (INT32_MAX)
        stop_tol                  = 1e-4        % Stopping tolerance
        time_limit                = 3600.0      % Time limit in seconds
        device_number             = 0           % GPU device ID
        check_iter                = 150         % Check convergence every N iterations
        use_Ruiz_scaling          = true        % Use Ruiz scaling
        use_Pock_Chambolle_scaling = true       % Use Pock-Chambolle scaling
        use_bc_scaling            = true        % Use bound constraint scaling
    end
    
    methods
        function obj = Parameters()
            % Parameters - Constructor with default values
            %
            % Creates a Parameters object with default solver settings.
            % All properties can be modified after construction.
        end
        
        function s = toStruct(obj)
            % toStruct - Convert Parameters object to struct for MEX interface
            %
            % Returns:
            %   s - Structure with all parameter fields
            
            s.max_iter = obj.max_iter;
            s.stop_tol = obj.stop_tol;
            s.time_limit = obj.time_limit;
            s.device_number = obj.device_number;
            s.check_iter = obj.check_iter;
            s.use_Ruiz_scaling = obj.use_Ruiz_scaling;
            s.use_Pock_Chambolle_scaling = obj.use_Pock_Chambolle_scaling;
            s.use_bc_scaling = obj.use_bc_scaling;
        end
        
        function disp(obj)
            % disp - Display parameter values
            
            fprintf('HPRLP Parameters:\n');
            fprintf('  max_iter:                   %d\n', obj.max_iter);
            fprintf('  stop_tol:                   %.2e\n', obj.stop_tol);
            fprintf('  time_limit:                 %.1f s\n', obj.time_limit);
            fprintf('  device_number:              %d\n', obj.device_number);
            fprintf('  check_iter:                 %d\n', obj.check_iter);
            fprintf('  use_Ruiz_scaling:           %d\n', obj.use_Ruiz_scaling);
            fprintf('  use_Pock_Chambolle_scaling: %d\n', obj.use_Pock_Chambolle_scaling);
            fprintf('  use_bc_scaling:             %d\n', obj.use_bc_scaling);
        end
    end
end
