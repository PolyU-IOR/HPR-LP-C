classdef Model < handle
    %MODEL Represents an LP problem model for HPRLP solver
    %
    % This class provides a high-level interface to the HPRLP solver using
    % a model-based API. Models store the LP problem data and can be solved
    % multiple times with different parameters.
    %
    % CONSTRUCTION:
    %   model = hprlp.Model.from_arrays(A, AL, AU, l, u, c)
    %       Create a model from constraint matrix and vectors
    %
    %   model = hprlp.Model.from_mps(filename)
    %       Create a model from an MPS file
    %
    % METHODS:
    %   result = model.solve()
    %       Solve with default parameters
    %
    %   result = model.solve(params)
    %       Solve with custom parameters
    %
    % PROPERTIES (Read-only):
    %   m              - Number of constraints
    %   n              - Number of variables
    %   obj_constant   - Constant term in objective
    %
    % EXAMPLE:
    %   % Create model from arrays
    %   A = sparse([1 2; 3 1]);
    %   AL = [-inf; -inf];
    %   AU = [10; 12];
    %   l = [0; 0];
    %   u = [inf; inf];
    %   c = [-3; -5];
    %   
    %   model = hprlp.Model.from_arrays(A, AL, AU, l, u, c);
    %   result = model.solve();
    %   
    %   fprintf('Optimal value: %.6f\n', result.primal_obj);
    %   fprintf('Solution: [%.6f, %.6f]\n', result.x(1), result.x(2));
    %
    % See also hprlp.Parameters, hprlp.Result
    
    properties (SetAccess = private, GetAccess = public)
        m              % Number of constraints
        n              % Number of variables  
        obj_constant   % Constant term in objective
    end
    
    properties (Access = private)
        handle_        % uint64 handle to C model
    end
    
    methods (Static)
        function obj = from_arrays(A, AL, AU, l, u, c, varargin)
            %FROM_ARRAYS Create model from constraint matrix and vectors
            %
            % model = hprlp.Model.from_arrays(A, AL, AU, l, u, c)
            % model = hprlp.Model.from_arrays(A, AL, AU, l, u, c, 'obj_constant', val)
            %
            % Represents the LP:
            %   minimize    c'*x + obj_constant
            %   subject to  AL <= A*x <= AU
            %               l <= x <= u
            %
            % INPUTS:
            %   A            - Constraint matrix (m x n sparse matrix)
            %   AL           - Lower bounds on constraints (m x 1)
            %   AU           - Upper bounds on constraints (m x 1)
            %   l            - Lower bounds on variables (n x 1)
            %   u            - Upper bounds on variables (n x 1)
            %   c            - Objective coefficients (n x 1)
            %   obj_constant - (Optional) Constant term in objective (default: 0)
            %
            % OUTPUTS:
            %   model - Model object
            
            % Parse optional arguments
            p = inputParser;
            addParameter(p, 'obj_constant', 0.0, @isnumeric);
            parse(p, varargin{:});
            obj_constant = p.Results.obj_constant;
            
            % Validate inputs
            if ~issparse(A)
                error('HPRLP:InvalidInput', 'Matrix A must be sparse');
            end
            
            [m, n] = size(A);
            
            if length(AL) ~= m
                error('HPRLP:InvalidInput', 'AL must have length m');
            end
            if length(AU) ~= m
                error('HPRLP:InvalidInput', 'AU must have length m');
            end
            if length(l) ~= n
                error('HPRLP:InvalidInput', 'l must have length n');
            end
            if length(u) ~= n
                error('HPRLP:InvalidInput', 'u must have length n');
            end
            if length(c) ~= n
                error('HPRLP:InvalidInput', 'c must have length n');
            end
            
            % Ensure vectors are column vectors
            AL = AL(:);
            AU = AU(:);
            l = l(:);
            u = u(:);
            c = c(:);
            
            % Call MEX function to create model
            handle = hprlp_mex('create_model_from_arrays', A, AL, AU, l, u, c);
            
            % Get model info (m, n, obj_constant) from the C model
            info = hprlp_mex('get_model_info', handle);
            
            % Create object
            obj = hprlp.Model();
            obj.handle_ = handle;
            obj.m = info.m;
            obj.n = info.n;
            obj.obj_constant = obj_constant;  % Use the provided value, not from model
        end
        
        function obj = from_mps(filename)
            %FROM_MPS Create model from MPS file
            %
            % model = hprlp.Model.from_mps(filename)
            %
            % INPUTS:
            %   filename - Path to MPS file
            %
            % OUTPUTS:
            %   model - Model object
            
            % Validate input
            if ~ischar(filename) && ~isstring(filename)
                error('HPRLP:InvalidInput', 'filename must be a string');
            end
            
            if ~isfile(filename)
                error('HPRLP:InvalidInput', 'MPS file not found: %s', filename);
            end
            
            % Call MEX function to create model
            handle = hprlp_mex('create_model_from_mps', filename);
            
            % Get model info (m, n, obj_constant) from the C model
            info = hprlp_mex('get_model_info', handle);
            
            % Create object
            obj = hprlp.Model();
            obj.handle_ = handle;
            obj.m = info.m;
            obj.n = info.n;
            obj.obj_constant = info.obj_constant;
        end
    end
    
    methods
        function result = solve(obj, params)
            %SOLVE Solve the LP model
            %
            % result = model.solve()
            %   Solve with default parameters
            %
            % result = model.solve(params)
            %   Solve with custom parameters
            %
            % INPUTS:
            %   params - (Optional) hprlp.Parameters object
            %
            % OUTPUTS:
            %   result - hprlp.Result object containing solution
            
            if obj.handle_ == 0
                error('HPRLP:InvalidModel', 'Model has been freed');
            end
            
            % Call MEX function
            if nargin < 2 || isempty(params)
                % Solve with default parameters
                result_struct = hprlp_mex('solve', obj.handle_);
            else
                % Convert Parameters object to struct
                if isa(params, 'hprlp.Parameters')
                    param_struct = params.toStruct();
                else
                    error('HPRLP:InvalidInput', 'params must be hprlp.Parameters object');
                end
                result_struct = hprlp_mex('solve', obj.handle_, param_struct);
            end
            
            % Create Result object
            result = hprlp.Result(result_struct);
            
            % Adjust objective value by constant
            result.primal_obj = result.primal_obj + obj.obj_constant;
        end
        
        function delete(obj)
            %DELETE Destructor - frees model memory
            %
            % This is called automatically when the object is destroyed
            
            if obj.handle_ ~= 0
                hprlp_mex('free_model', obj.handle_);
                obj.handle_ = 0;
            end
        end
    end
    
    methods (Access = private)
        function obj = Model()
            %MODEL Private constructor
            % Use from_arrays() or from_mps() to create models
            obj.handle_ = 0;
            obj.m = 0;
            obj.n = 0;
            obj.obj_constant = 0.0;
        end
    end
end
