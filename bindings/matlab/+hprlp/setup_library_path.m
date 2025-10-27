function setup_library_path()
% setup_library_path - Add HPRLP library to system path
%
% This function adds the HPRLP shared library directory to the system
% library path. Call this function if you get an error about libhprlp.so
% not being found.
%
% Usage:
%   hprlp.setup_library_path()
%
% This function sets LD_LIBRARY_PATH on Linux or DYLD_LIBRARY_PATH on macOS.

    % Get the library directory
    matlab_dir = fileparts(fileparts(mfilename('fullpath')));
    lib_dir = fullfile(matlab_dir, '..', '..', 'lib');
    lib_dir = char(java.io.File(lib_dir).getCanonicalPath());
    
    % Check if library exists
    if isunix && ~ismac
        lib_file = fullfile(lib_dir, 'libhprlp.so');
        env_var = 'LD_LIBRARY_PATH';
    elseif ismac
        lib_file = fullfile(lib_dir, 'libhprlp.dylib');
        env_var = 'DYLD_LIBRARY_PATH';
    else
        error('HPRLP:UnsupportedPlatform', 'Windows is not currently supported');
    end
    
    if ~exist(lib_file, 'file')
        error('HPRLP:LibraryNotFound', ...
              'Library not found: %s\nPlease build the library first: cd ../..; make', ...
              lib_file);
    end
    
    % Get current library path
    current_path = getenv(env_var);
    
    % Check if already in path
    if contains(current_path, lib_dir)
        fprintf('Library directory already in %s\n', env_var);
        return;
    end
    
    % Add to library path
    if isempty(current_path)
        new_path = lib_dir;
    else
        new_path = [lib_dir ':' current_path];
    end
    
    setenv(env_var, new_path);
    
    fprintf('✓ Added to %s: %s\n', env_var, lib_dir);
    fprintf('  Note: This setting only persists for the current MATLAB session.\n');
    fprintf('  To make it permanent, add this to your startup.m:\n');
    fprintf('    setenv(''%s'', ''%s:%s'');\n', env_var, lib_dir, current_path);
end
