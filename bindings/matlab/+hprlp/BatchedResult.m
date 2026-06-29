classdef BatchedResult
    %BatchedResult Results from batched HPRLP solves sharing one matrix A.

    properties
        status
        residuals
        primal_obj
        gap
        iter
        time
        setup_time
        solve_time
        power_time
        x
        y
        z
    end

    methods
        function obj = BatchedResult(result_struct)
            if nargin > 0
                obj.status = result_struct.status;
                obj.residuals = result_struct.residuals;
                obj.primal_obj = result_struct.primal_obj;
                obj.gap = result_struct.gap;
                obj.iter = result_struct.iter;
                obj.time = result_struct.time;
                obj.setup_time = result_struct.setup_time;
                obj.solve_time = result_struct.solve_time;
                obj.power_time = result_struct.power_time;
                obj.x = result_struct.x;
                obj.y = result_struct.y;
                obj.z = result_struct.z;
            end
        end

        function tf = isOptimal(obj)
            tf = all(strcmp(obj.status, 'OPTIMAL'));
        end

        function disp(obj)
            B = numel(obj.status);
            fprintf('HPRLP Batched Results:\n');
            fprintf('  Batch size:    %d\n', B);
            fprintf('  Max residual:  %.6e\n', max(obj.residuals));
            fprintf('  Time:          %.4f s\n', obj.time);
            for k = 1:B
                fprintf('  [%d] status=%s obj=%.12e residual=%.6e iter=%d\n', ...
                    k, obj.status{k}, obj.primal_obj(k), obj.residuals(k), obj.iter(k));
            end
        end
    end
end
