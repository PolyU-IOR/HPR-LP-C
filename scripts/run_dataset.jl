#!/usr/bin/env julia

const DEFAULT_HPRLP_ENVIRONMENT = (
    "HPRLP_ENABLE_COMPRESSIBLE_MEMORY" => "1",
    "HPRLP_USE_ROW_REDUCTION" => "1",
    "HPRLP_USE_ROW_COMPRESSED_AUTOTUNE" => "1",
    "HPRLP_USE_REDUCED_COMPRESSED_AUTOTUNE" => "1",
    "HPRLP_DEFER_REDUCED_EMPTY_ROWS_TO_CHECK" => "1",
    "HPRLP_USE_REDUCED_NONEMPTY_CUSPARSE" => "1",
    "HPRLP_REDUCED_RESET_MASK_ON_RESTART" => "1",
    "HPRLP_REDUCED_RESTART_MASK_MIN_RECOVERY" => "0.25",
    "HPRLP_REDUCED_RESTART_MASK_MIN_SAVED_COLUMNS" => "25000",
    "HPRLP_REDUCED_RESTART_MASK_MIN_CURRENT_COLUMNS" => "95000",
)
for (name, value) in DEFAULT_HPRLP_ENVIRONMENT
    get!(ENV, name, value)
end

import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "bindings", "julia", "package"); io=devnull)

using Dates
using Distributed
using HDF5
using Printf

module HPRLPBatchAPI
include(joinpath(@__DIR__, "..", "bindings", "julia", "package", "src", "wrapper.jl"))
end
using .HPRLPBatchAPI

const API = HPRLPBatchAPI

struct C_sparseMatrix
    row::Int32
    col::Int32
    numElements::Int32
    colIndex::Ptr{Int32}
    rowPtr::Ptr{Int32}
    value::Ptr{Float64}
end

struct C_LP_info_cpu
    m::Int32
    n::Int32
    A::Ptr{C_sparseMatrix}
    AL::Ptr{Float64}
    AU::Ptr{Float64}
    c::Ptr{Float64}
    l::Ptr{Float64}
    u::Ptr{Float64}
    obj_constant::Float64
end

mutable struct Options
    device::Int32
    devices::Vector{Int32}
    max_iter::Int32
    tol::Float64
    time_limit::Float64
    check_iter::Int32
    cusparse_spmv::Bool
    autotune_verbose::Bool
    enable_progress_monitor::Bool
    enable_progress_control::Bool
    enable_sigma_rebalance_restart::Bool
    use_progress_restart_guard::Bool
    restart_cooldown_checks::Int32
    debug_restart::Bool
    debug_sigma::Bool
    fixed_sigma::Float64
    cr::Bool
    ruiz::Bool
    pock::Bool
    bc::Bool
    presolver::Symbol
    gpu_folding::Bool
    use_reduced_matrix::Bool
    auto_reduced_compression_policy::Bool
    print_debug_info::Bool
    specified_parameter_mask::UInt64
    resume::Bool
end

Options() = Options(
    0, Int32[], typemax(Int32), 1e-6, 1000.0, 150,
    false, false,
    true, true, true, false, 0, false, false, NaN,
    true, true, true, true, :gpu, true, true, false,
    false, UInt64(0), true,
)

const PARAM_MAX_ITER = UInt64(1) << 0
const PARAM_STOP_TOL = UInt64(1) << 1
const PARAM_TIME_LIMIT = UInt64(1) << 2
const PARAM_DEVICE = UInt64(1) << 3
const PARAM_CHECK_ITER = UInt64(1) << 4
const PARAM_CUSPARSE_SPMV = UInt64(1) << 5
const PARAM_AUTOTUNE_VERBOSE = UInt64(1) << 6
const PARAM_PROGRESS_MONITOR = UInt64(1) << 7
const PARAM_PROGRESS_CONTROL = UInt64(1) << 8
const PARAM_SIGMA_REBALANCE = UInt64(1) << 9
const PARAM_RESTART_GUARD = UInt64(1) << 10
const PARAM_RESTART_COOLDOWN = UInt64(1) << 11
const PARAM_DEBUG_RESTART = UInt64(1) << 12
const PARAM_DEBUG_SIGMA = UInt64(1) << 13
const PARAM_FIXED_SIGMA = UInt64(1) << 14
const PARAM_CR_SCALING = UInt64(1) << 15
const PARAM_RUIZ_SCALING = UInt64(1) << 16
const PARAM_POCK_SCALING = UInt64(1) << 17
const PARAM_BC_SCALING = UInt64(1) << 18
const PARAM_PRESOLVER = UInt64(1) << 19
const PARAM_GPU_FOLDING = UInt64(1) << 20
const PARAM_REDUCED_MATRIX = UInt64(1) << 21
const PARAM_AUTO_MEMORY_POLICY = UInt64(1) << 22
const PARAM_PRINT_DEBUG_INFO = UInt64(1) << 23

const CSV_HEADER = [
    "instance_name", "status", "residuals", "primal_obj", "gap",
    "read_time", "total_time", "presolve_time", "setup_time",
    "scaling_time", "analyze_time", "power_iteration_time", "solve_time",
    "folding_time",
    "time4", "time6", "time8", "iter4", "iter6", "iter8", "total_iter",
    "m", "n", "nnz",
]

function usage(io::IO=stdout)
    println(io, "Usage: julia scripts/run_dataset.jl <lp_folder> [result_folder] [options]")
    println(io)
    println(io, "Options:")
    println(io, "  --devices <id,id,...>        CUDA devices; next job goes to next free device")
    println(io, "  --device <id>                Single-device fallback (default: 0)")
    println(io, "  --max-iter <N>               Max iterations (default: INT32_MAX)")
    println(io, "  --tol <eps>                  Stopping tolerance (default: 1e-6)")
    println(io, "  --time-limit <sec>           Time limit in seconds (default: 1000)")
    println(io, "  --check-iter <N>             Check interval (default: 150)")
    println(io, "  --cusparse-spmv <true|false>")
    println(io, "  --autotune-verbose <true|false>")
    println(io, "  --progress-monitor <true|false>")
    println(io, "  --progress-control <true|false>")
    println(io, "  --sigma-rebalance-restart <true|false>")
    println(io, "  --progress-restart-guard <true|false>")
    println(io, "  --restart-cooldown-checks <N>")
    println(io, "  --debug-restart <true|false>")
    println(io, "  --debug-sigma <true|false>")
    println(io, "  --print-debug-info <true|false> Detailed solver logging (default: false)")
    println(io, "  --fixed-sigma <value|nan>")
    println(io, "  --cr <true|false>")
    println(io, "  --ruiz <true|false>")
    println(io, "  --pock <true|false>")
    println(io, "  --bc <true|false>")
    println(io, "  --presolver <pslp|gpu|none>  Select presolver backend (default: gpu)")
    println(io, "  --gpu-folding <true|false>   Enable GPU-presolver folding (default: true)")
    println(io, "  --reduced-matrix <true|false> Enable adaptive row/column reduction in manual mode (default: true)")
    println(io, "  --auto-memory-policy <true|false> Couple reduced/compression from presolved dimensions (default: false)")
    println(io, "  --resume                     Skip existing CSV entries (default)")
    println(io, "  --no-resume                  Start a new CSV")
    println(io, "  -h, --help")
end

function parse_bool(value::String)
    lower = lowercase(value)
    lower in ("true", "1") && return true
    lower in ("false", "0") && return false
    error("Expected true or false, got: $value")
end

function parse_devices(value::String)
    text = strip(value)
    if startswith(text, "[") && endswith(text, "]")
        text = text[2:end-1]
    end
    devices = Int32[]
    for part in split(text, ',')
        isempty(strip(part)) || push!(devices, parse(Int32, strip(part)))
    end
    isempty(devices) && error("--devices must contain at least one CUDA device id")
    any(<(0), devices) && error("CUDA device ids must be nonnegative")
    length(unique(devices)) == length(devices) ||
        error("--devices must not contain duplicate CUDA device ids")
    return devices
end

function parse_args(args)
    options = Options()
    positional = String[]
    i = 1
    while i <= length(args)
        arg = args[i]
        if arg in ("-h", "--help")
            usage()
            exit(0)
        elseif arg == "--resume"
            options.resume = true
            i += 1
        elseif arg == "--no-resume"
            options.resume = false
            i += 1
        elseif startswith(arg, "--")
            i == length(args) && error("Missing value for option: $arg")
            value = args[i + 1]
            if arg == "--devices"
                options.devices = parse_devices(value)
                options.specified_parameter_mask |= PARAM_DEVICE
            elseif arg == "--device"
                options.device = parse(Int32, value)
                options.device >= 0 || error("CUDA device id must be nonnegative")
                options.specified_parameter_mask |= PARAM_DEVICE
            elseif arg == "--max-iter"
                options.max_iter = parse(Int32, value)
                options.specified_parameter_mask |= PARAM_MAX_ITER
            elseif arg == "--tol"
                options.tol = parse(Float64, value)
                options.specified_parameter_mask |= PARAM_STOP_TOL
            elseif arg == "--time-limit"
                options.time_limit = parse(Float64, value)
                options.specified_parameter_mask |= PARAM_TIME_LIMIT
            elseif arg == "--check-iter"
                options.check_iter = parse(Int32, value)
                options.specified_parameter_mask |= PARAM_CHECK_ITER
            elseif arg == "--cusparse-spmv"
                options.cusparse_spmv = parse_bool(value)
                options.specified_parameter_mask |= PARAM_CUSPARSE_SPMV
            elseif arg == "--autotune-verbose"
                options.autotune_verbose = parse_bool(value)
                options.specified_parameter_mask |= PARAM_AUTOTUNE_VERBOSE
            elseif arg == "--progress-monitor"
                options.enable_progress_monitor = parse_bool(value)
                options.specified_parameter_mask |= PARAM_PROGRESS_MONITOR
            elseif arg == "--progress-control"
                options.enable_progress_control = parse_bool(value)
                options.specified_parameter_mask |= PARAM_PROGRESS_CONTROL
            elseif arg == "--sigma-rebalance-restart"
                options.enable_sigma_rebalance_restart = parse_bool(value)
                options.specified_parameter_mask |= PARAM_SIGMA_REBALANCE
            elseif arg == "--progress-restart-guard"
                options.use_progress_restart_guard = parse_bool(value)
                options.specified_parameter_mask |= PARAM_RESTART_GUARD
            elseif arg == "--restart-cooldown-checks"
                options.restart_cooldown_checks = parse(Int32, value)
                options.specified_parameter_mask |= PARAM_RESTART_COOLDOWN
            elseif arg == "--debug-restart"
                options.debug_restart = parse_bool(value)
                options.specified_parameter_mask |= PARAM_DEBUG_RESTART
            elseif arg == "--debug-sigma"
                options.debug_sigma = parse_bool(value)
                options.specified_parameter_mask |= PARAM_DEBUG_SIGMA
            elseif arg == "--print-debug-info"
                options.print_debug_info = parse_bool(value)
                options.specified_parameter_mask |= PARAM_PRINT_DEBUG_INFO
            elseif arg == "--fixed-sigma"
                options.fixed_sigma = parse(Float64, value)
                options.specified_parameter_mask |= PARAM_FIXED_SIGMA
            elseif arg == "--cr"
                options.cr = parse_bool(value)
                options.specified_parameter_mask |= PARAM_CR_SCALING
            elseif arg == "--ruiz"
                options.ruiz = parse_bool(value)
                options.specified_parameter_mask |= PARAM_RUIZ_SCALING
            elseif arg == "--pock"
                options.pock = parse_bool(value)
                options.specified_parameter_mask |= PARAM_POCK_SCALING
            elseif arg == "--bc"
                options.bc = parse_bool(value)
                options.specified_parameter_mask |= PARAM_BC_SCALING
            elseif arg == "--presolver"
                backend = Symbol(lowercase(value))
                backend in (:pslp, :gpu, :none) ||
                    error("Presolver must be pslp, gpu, or none")
                options.presolver = backend
                options.specified_parameter_mask |= PARAM_PRESOLVER
            elseif arg == "--gpu-folding"
                options.gpu_folding = parse_bool(value)
                options.specified_parameter_mask |= PARAM_GPU_FOLDING
            elseif arg == "--reduced-matrix"
                options.use_reduced_matrix = parse_bool(value)
                options.specified_parameter_mask |= PARAM_REDUCED_MATRIX
            elseif arg == "--auto-memory-policy"
                options.auto_reduced_compression_policy = parse_bool(value)
                options.specified_parameter_mask |= PARAM_AUTO_MEMORY_POLICY
            else
                error("Unknown option: $arg")
            end
            i += 2
        else
            push!(positional, arg)
            i += 1
        end
    end
    isempty(positional) && error("MPS folder is required")
    length(positional) > 2 && error("Too many positional arguments")
    isempty(options.devices) && push!(options.devices, options.device)
    options.device = first(options.devices)
    return positional[1], length(positional) == 2 ? positional[2] : "./results", options
end

function c_parameters(options::Options)
    presolver = options.presolver === :pslp ? Int32(0) :
                options.presolver === :gpu ? Int32(1) : Int32(2)
    use_presolve = options.presolver !== :none
    return API.C_HPRLP_parameters(
        options.max_iter, options.tol, options.time_limit,
        options.device, options.check_iter,
        options.cusparse_spmv, options.autotune_verbose,
        options.enable_progress_monitor,
        options.enable_progress_control,
        options.enable_sigma_rebalance_restart,
        options.use_progress_restart_guard,
        options.restart_cooldown_checks,
        options.debug_restart, options.debug_sigma, options.fixed_sigma,
        options.cr, options.ruiz, options.pock, options.bc,
        use_presolve, options.gpu_folding, presolver,
        options.use_reduced_matrix,
        options.auto_reduced_compression_policy,
        options.print_debug_info,
        options.specified_parameter_mask,
    )
end

is_lp_file(path::String) = endswith(lowercase(path), ".mps") ||
                           endswith(lowercase(path), ".mps.gz") ||
                           endswith(lowercase(path), ".h5") ||
                           endswith(lowercase(path), ".hdf5")

function instance_name(path::String)
    name = basename(path)
    lower = lowercase(name)
    if endswith(lower, ".mps.gz")
        return name[1:end-7]
    elseif endswith(lower, ".hdf5")
        return name[1:end-5]
    elseif endswith(lower, ".h5")
        return name[1:end-3]
    elseif endswith(lower, ".mps")
        return name[1:end-4]
    end
    return name
end

function instance_log_name(path::String)
    safe = replace(instance_name(path), r"[^A-Za-z0-9._-]" => "_")
    return (isempty(safe) ? "instance" : safe) * ".log"
end

function csv_field(value)
    text = string(value)
    return "\"" * replace(text, "\"" => "\"\"") * "\""
end

function write_csv_row(io::IO, values)
    println(io, join(csv_field.(values), ","))
    flush(io)
end

function parse_csv_line(line::String)
    text = chomp(line)
    fields = String[]
    field = IOBuffer()
    quoted = false
    i = firstindex(text)
    while i <= lastindex(text)
        char = text[i]
        if quoted
            if char == '"'
                next = nextind(text, i)
                if next <= lastindex(text) && text[next] == '"'
                    write(field, '"')
                    i = nextind(text, next)
                    continue
                end
                quoted = false
            else
                write(field, char)
            end
        elseif char == '"'
            quoted = true
        elseif char == ','
            push!(fields, String(take!(field)))
        elseif char != '\r'
            write(field, char)
        end
        i = nextind(text, i)
    end
    push!(fields, String(take!(field)))
    return fields
end

function summary_row_from_values(values)
    length(values) == length(CSV_HEADER) || return nothing
    numeric(field) = tryparse(Float64, values[findfirst(==(field), CSV_HEADER)])
    total_iter = numeric("total_iter")
    read_time = numeric("read_time")
    total_time = numeric("total_time")
    presolve_time = numeric("presolve_time")
    setup_time = numeric("setup_time")
    scaling_time = numeric("scaling_time")
    analyze_time = numeric("analyze_time")
    power_iteration_time = numeric("power_iteration_time")
    solve_time = numeric("solve_time")
    folding_time = numeric("folding_time")
    any(isnothing, (
        total_iter, read_time, total_time, presolve_time, setup_time,
        scaling_time, analyze_time, power_iteration_time, solve_time,
        folding_time,
    )) && return nothing
    status = values[findfirst(==("status"), CSV_HEADER)]
    return (
        total_iter=total_iter, read_time=read_time,
        total_time=total_time, presolve_time=presolve_time,
        setup_time=setup_time, scaling_time=scaling_time,
        analyze_time=analyze_time,
        power_iteration_time=power_iteration_time,
        solve_time=solve_time, folding_time=folding_time,
        solved=(status == "OPTIMAL" || startswith(status, "REDUCED_OPTIMAL_")),
    )
end

function read_existing_results(csv_path::String)
    rows = Vector{Vector{String}}()
    done = Set{String}()
    summaries = NamedTuple[]
    (!isfile(csv_path) || filesize(csv_path) == 0) && return rows, done, summaries
    lines = readlines(csv_path)
    isempty(lines) && return rows, done, summaries
    parse_csv_line(lines[1]) == CSV_HEADER ||
        error("Existing CSV header does not match: $csv_path")
    for line in lines[2:end]
        isempty(strip(line)) && continue
        values = parse_csv_line(line)
        length(values) == length(CSV_HEADER) || continue
        name = values[1]
        name in ("SGM10", "solved") && continue
        push!(rows, values)
        push!(done, name)
        summary = summary_row_from_values(values)
        summary === nothing || push!(summaries, summary)
    end
    return rows, done, summaries
end

function shifted_geomean(values; shift=10.0)
    isempty(values) && return 0.0
    result = exp(sum(log(value + shift) for value in values) / length(values)) - shift
    return abs(result) < 1e-12 ? 0.0 : result
end

function write_summary_rows(csv::IO, rows, solved_count::Int)
    isempty(rows) && return

    summary = fill("", length(CSV_HEADER))
    summary[1] = "SGM10"
    for field in (:total_iter, :read_time, :total_time, :presolve_time,
                  :setup_time, :scaling_time, :analyze_time,
                  :power_iteration_time, :solve_time, :folding_time)
        column = findfirst(==(String(field)), CSV_HEADER)
        values = [getproperty(row, field) for row in rows]
        summary[column] = string(shifted_geomean(values))
    end
    write_csv_row(csv, summary)

    solved = fill("", length(CSV_HEADER))
    solved[1] = "solved"
    solved[findfirst(==("total_time"), CSV_HEADER)] = string(solved_count)
    solved[findfirst(==("solve_time"), CSV_HEADER)] = string(solved_count)
    write_csv_row(csv, solved)
end

function status_string(status::NTuple{64, UInt8})
    bytes = collect(status)
    stop = findfirst(==(0x00), bytes)
    return String(stop === nothing ? bytes : bytes[1:stop-1])
end

function model_dimensions(model_ptr::Ptr{Cvoid})
    model = unsafe_load(Ptr{C_LP_info_cpu}(model_ptr))
    matrix = unsafe_load(model.A)
    return Int(model.m), Int(model.n), Int(matrix.numElements)
end

function create_model(path::String)
    lower = lowercase(path)
    if endswith(lower, ".h5") || endswith(lower, ".hdf5")
        read_start = time()
        data = h5open(path, "r") do file
            version = Int32(read(file, "schema_version"))
            version == 1 || error("Unsupported LP HDF5 schema version $version")
            matrix_size = Vector{Int64}(read(file, "A/size"))
            length(matrix_size) == 2 || error("Invalid A/size dataset")
            m, n = Int.(matrix_size)
            colptr = Vector{Int32}(read(file, "A/colptr"))
            rowval = Vector{Int32}(read(file, "A/rowval"))
            nzval = Vector{Float64}(read(file, "A/nzval"))
            c = Vector{Float64}(read(file, "c"))
            AL = Vector{Float64}(read(file, "AL"))
            AU = Vector{Float64}(read(file, "AU"))
            l = Vector{Float64}(read(file, "l"))
            u = Vector{Float64}(read(file, "u"))
            obj_constant = Float64(read(file, "obj_constant"))
            return m, n, colptr, rowval, nzval, c, AL, AU, l, u, obj_constant
        end
        m, n, colptr, rowval, nzval, c, AL, AU, l, u, obj_constant = data
        length(colptr) == n + 1 || error("Invalid A/colptr length")
        length(rowval) == length(nzval) || error("Invalid sparse arrays")
        length(c) == n || error("Invalid c length")
        length(l) == n || error("Invalid l length")
        length(u) == n || error("Invalid u length")
        length(AL) == m || error("Invalid AL length")
        length(AU) == m || error("Invalid AU length")
        first(colptr) == 1 || error("A/colptr must be 1-based")
        last(colptr) == length(nzval) + 1 || error("Invalid final A/colptr")
        all(index -> 1 <= index <= m, rowval) ||
            error("A/rowval contains an out-of-range row index")
        colptr .-= 1
        rowval .-= 1
        model_ptr = API.c_create_model_from_arrays_with_obj_constant(
            m, n, length(nzval), colptr, rowval, nzval,
            AL, AU, l, u, c, obj_constant, true,
        )
        return model_ptr, time() - read_start
    end
    return API.c_create_model_from_mps(path)
end

function log_message(io::IO, message)
    stamped = "[$(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))] $message"
    println(io, stamped)
    flush(io)
    println(stamped)
end

function capture_solver_output(solve_call, log::IO)
    return redirect_stdout(log) do
        redirect_stderr(log) do
            solve_call()
        end
    end
end

function failure_values(path::String, status::String)
    return Any[instance_name(path), status, fill("", length(CSV_HEADER) - 2)...]
end

function solve_one_worker(
    index::Int,
    total::Int,
    path::String,
    log_path::String,
    options::Options,
)
    read_time = 0.0
    model_ptr = C_NULL
    result = nothing
    values = failure_values(path, "SOLVE_ERROR")
    summary = nothing
    success = false
    status = "SOLVE_ERROR"

    mkpath(dirname(log_path))
    open(log_path, "a") do log
        flush_timer = Timer(_ -> try flush(log) catch end, 5.0; interval=5.0)
        try
            ENV["HPRLP_PRINT_DEBUG_INFO"] = options.print_debug_info ? "1" : "0"
            println(log, "Reading file $path")
            flush(log)
            model_ptr, read_time = create_model(path)
            @printf(log, "Reading time: %.2fs\n", read_time)
            flush(log)
            if options.print_debug_info
                println(log)
                println(log, "= "^50)
                log_message(log, "[$index/$total] Processing $(instance_name(path)) on device $(options.device)")
            end
            if model_ptr == C_NULL
                status = "READ_ERROR"
                values = failure_values(path, status)
                log_message(log, "  Failed to read model")
                return (values=values, summary=summary, success=false, status=status)
            end

            m, n, nnz = model_dimensions(model_ptr)
            options.print_debug_info &&
                log_message(log, "  Dimensions: m=$m, n=$n, nnz=$nnz")
            result = capture_solver_output(log) do
                API.c_solve_model(model_ptr, c_parameters(options))
            end
            timing = result.timing
            status = status_string(result.status)
            values = Any[
                instance_name(path), status,
                result.residuals, result.primal_obj, result.gap,
                read_time, timing.total_time, timing.presolve_time,
                timing.setup_time, timing.scaling_time, timing.analyze_time,
                timing.power_iteration_time, timing.solve_time,
                result.folding_time, result.time4, result.time6, result.time8,
                result.iter4, result.iter6, result.iter8, result.iter,
                m, n, nnz,
            ]
            summary = (
                total_iter=Float64(result.iter), read_time=read_time,
                total_time=timing.total_time, presolve_time=timing.presolve_time,
                setup_time=timing.setup_time, scaling_time=timing.scaling_time,
                analyze_time=timing.analyze_time,
                power_iteration_time=timing.power_iteration_time,
                solve_time=timing.solve_time, folding_time=result.folding_time,
                solved=(status == "OPTIMAL" || startswith(status, "REDUCED_OPTIMAL_")),
            )
            success = true
            options.print_debug_info &&
                log_message(log, "  Status: $status; total_time=$(timing.total_time) sec")
        catch err
            status = model_ptr == C_NULL ? "READ_ERROR" : "SOLVE_ERROR"
            values = failure_values(path, status)
            log_message(log, "  ERROR: $(sprint(showerror, err, catch_backtrace()))")
        finally
            close(flush_timer)
            try
                flush(log)
            catch
            end
            if result !== nothing
                API.c_free_results(result.x, result.y, result.z)
            end
            if model_ptr != C_NULL
                API.c_free_model(model_ptr)
            end
        end
    end
    GC.gc(false)
    return (values=values, summary=summary, success=success, status=status)
end

function record_worker_failure(path::String, log_path::String, device::Int32, err)
    message = sprint(showerror, err)
    open(log_path, "a") do log
        log_message(log, "Worker on device $device failed: $message")
    end
    return (
        values=failure_values(path, "WORKER_ERROR"),
        summary=nothing,
        success=false,
        status="WORKER_ERROR",
    )
end

function spawn_dataset_workers(devices::Vector{Int32}, script_path::String)
    workers = Int[]
    try
        for _ in devices
            worker = only(addprocs(1))
            Distributed.remotecall_eval(Main, worker, :(include($script_path)))
            push!(workers, worker)
        end
    catch
        isempty(workers) || rmprocs(workers...)
        rethrow()
    end
    return workers
end

function main(args)
    lp_folder, result_folder, options = parse_args(args)
    isdir(lp_folder) || error("LP folder does not exist: $lp_folder")
    mkpath(result_folder)

    files = sort(filter(is_lp_file, readdir(lp_folder; join=true)))
    isempty(files) &&
        error("No .mps, .mps.gz, .h5, or .hdf5 files found in $lp_folder")

    csv_path = joinpath(result_folder, "results.csv")
    log_path = joinpath(result_folder, "batch_solve.log")
    logs_dir = joinpath(result_folder, "logs")
    mkpath(logs_dir)

    old_rows, done_names, summary_rows = options.resume ?
        read_existing_results(csv_path) :
        (Vector{Vector{String}}(), Set{String}(), NamedTuple[])
    pending = Tuple{Int,String}[]
    for (index, path) in enumerate(files)
        name = instance_name(path)
        if name in done_names
            println("The result of problem exists: $name")
        else
            push!(pending, (index, path))
        end
    end

    errors = 0
    open(csv_path, "w") do csv
        write_csv_row(csv, CSV_HEADER)
        foreach(row -> write_csv_row(csv, row), old_rows)

        open(log_path, "a") do log
            log_message(log, "HPRLP Julia multi-device batch solver")
            log_message(log, "LP folder: $lp_folder")
            log_message(log, "Result folder: $result_folder")
            log_message(log, "Dataset devices: $(join(options.devices, ','))")
            log_message(log, "Total instances: $(length(files)); pending: $(length(pending))")
            log_message(log, "Presolver: $(options.presolver)")
            log_message(log, "Automatic reduced/compression policy: $(options.auto_reduced_compression_policy)")
            log_message(log, "Instance logs: $logs_dir")

            if !isempty(pending)
                worker_ids = spawn_dataset_workers(options.devices, abspath(@__FILE__))
                job_lock = ReentrantLock()
                next_job = Ref(1)
                completed = Channel{Any}(length(pending))
                worker_tasks = Task[]
                try
                    for (worker, device) in zip(worker_ids, options.devices)
                        task = @async begin
                            while true
                                job = lock(job_lock) do
                                    if next_job[] > length(pending)
                                        nothing
                                    else
                                        assigned = pending[next_job[]]
                                        next_job[] += 1
                                        assigned
                                    end
                                end
                                job === nothing && break
                                index, path = job
                                local_options = deepcopy(options)
                                local_options.device = device
                                local_options.devices = Int32[device]
                                instance_log = joinpath(logs_dir, instance_log_name(path))
                                println("Assigning $(instance_name(path)) to GPU device $device (worker $worker)")
                                result = try
                                    remotecall_fetch(
                                        solve_one_worker, worker,
                                        index, length(files), path, instance_log,
                                        local_options,
                                    )
                                catch err
                                    record_worker_failure(path, instance_log, device, err)
                                end
                                put!(completed, (result=result, device=device))
                            end
                        end
                        push!(worker_tasks, task)
                    end

                    for _ in eachindex(pending)
                        completed_item = take!(completed)
                        result = completed_item.result
                        write_csv_row(csv, result.values)
                        if result.summary === nothing
                            errors += 1
                        else
                            push!(summary_rows, result.summary)
                        end
                        log_message(
                            log,
                            "Completed $(result.values[1]) on device $(completed_item.device) " *
                            "with status $(result.status)",
                        )
                    end
                    foreach(wait, worker_tasks)
                finally
                    for task in worker_tasks
                        istaskdone(task) || schedule(task, InterruptException(); error=true)
                    end
                    isempty(worker_ids) || rmprocs(worker_ids...)
                end
            end

            solved = count(row -> row.solved, summary_rows)
            write_summary_rows(csv, summary_rows, solved)
            log_message(log, "Batch complete: solved=$solved errors=$errors")
            log_message(log, "Results saved to: $csv_path")
        end
    end
    return errors == 0 ? 0 : 1
end

if abspath(PROGRAM_FILE) == @__FILE__
    try
        exit(main(ARGS))
    catch err
        println(stderr, "Error: ", sprint(showerror, err, catch_backtrace()))
        usage(stderr)
        exit(2)
    end
end
