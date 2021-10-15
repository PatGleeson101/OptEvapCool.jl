# Performing and plotting measurements during the simulation
using Statistics: mean
using RollingFunctions: rollmean, rolling
using Printf: @sprintf
using LaTeXStrings
using StatsPlots: plot, plot!, density, RGB, ylims, ylims!, savefig
# using StatsBase: fit, Histogram
using Dates: DateFormat, format, now
using DataFrames: DataFrame
using CSV

# Dummy function
null(args...) = nothing

# Compute translational kinetic energy
function avg_kinetic_energy(velocities, m)
    return 0.5 * m * mean( sum(velocities .^2, dims = 1) )
end

# Get a time-save string
filetime_fmt = DateFormat("yyyy-mm-dd-HH-MM-SS")
filetime() = format(now(), filetime_fmt)

# TODO: More versatile / customisable sensing

# Measure global quantities
struct GlobalSensor
    time :: Vector{Float64}
    ke :: Vector{Float64} # Per atom
    pe :: Vector{Float64} # Per atom
    F :: Vector{Float64}
    Nt :: Vector{Int64}
    cand :: Vector{Int64} # Total
    coll :: Vector{Int64} # Total
    Vsim :: Vector{Float64}
    n0 :: Vector{Float64} # Peak density

    function GlobalSensor(args...)
        # Check that all arrays have the same length
        if all(a -> length(a) == length(args[1]), args)
            return new(args...)
        else
            throw(ArgumentError("Sensor arrays must be equally sized."))
        end
    end
end

# Blank sensor
GlobalSensor() = GlobalSensor([zeros(0) for _ in 1:9]...)

# TODO: GridSensor

# Overload measurement
function measurer(sensor::GlobalSensor, p = 0, ωx = 0, ωy = 0, ωz = 0, centre = [0,0,0])
    ω_x, ω_y, ω_z = time_parametrize(ωx, ωy, ωz)
    centre = time_parametrize(centre)
    return (args...) -> measure(sensor, args..., p = p,
        ωx = ω_x, ωy = ω_y, ωz = ω_z, centre = centre)
end

# Save to CSV
function savecsv(sensor::GlobalSensor, filename)
    CSV.write(filename, DataFrame(
        time = sensor.time,
        ke = sensor.ke,
        pe = sensor.pe,
        F = sensor.F,
        Nt = sensor.Nt,
        cand = sensor.cand,
        coll = sensor.coll,
        Vsim = sensor.Vsim,
        n0 = sensor.n0
    ))
    return nothing
end

# Load from CSV
function loadsensor(filename)
    df = CSV.read(filename, DataFrame)
    return GlobalSensor(df.time, df.ke, df.pe, df.F, df.Nt,
                        df.cand, df.coll, df.Vsim, df.n0)
end

function trapped_cloud(cloud, conditions, T, ωx, ωy, ωz, p, centre)
    Nt = cloud.Nt
    species = conditions.species
    m = species.m

    N = 0
    positions = zeros(Float64, 3, Nt)
    velocities = zeros(Float64, 3, Nt)
    bound = - 2 * log(p)

    σx² = kB * T / (m * ωx^2)
    σy² = kB * T / (m * ωy^2)
    σz² = kB * T / (m * ωz^2)

    for atom in 1:Nt
        x, y, z = view(cloud.positions, :, atom) .- centre
        if (x^2 / σx² + y^2 / σy² + z^2 / σz²) < bound
            N += 1
            positions[1, N] = x
            positions[2, N] = y
            positions[3, N] = z
            velocities[:, N] = view(cloud.velocities, :, atom)
        end
    end
    positions = view(positions, :, 1:N)
    velocities = view(velocities, :, 1:N)

    return positions, velocities
end

# Perform measurements
function measure(sensor::GlobalSensor, cloud, conditions, cand_count, coll_count, t, n0, V;
                 p = 0, ωx = (t) -> 0, ωy = (t) -> 0, ωz = (t) -> 0, centre = (t) -> [0,0,0])
    
    T = (length(sensor.ke) > 0) ? last(sensor.time) : Inf
    positions, velocities = trapped_cloud(cloud, conditions, T, ωx(t), ωy(t), ωz(t), p, centre(t))
    #positions = view(cloud.positions, :, 1:cloud.Nt)
    #velocities = view(cloud.velocities, :, 1:cloud.Nt)

    species = conditions.species
    ke = avg_kinetic_energy(velocities, species.m)
    pe = mean( conditions.potential(positions, species, t) )

    push!(sensor.time, t)
    push!(sensor.ke, ke)
    push!(sensor.pe, pe)
    push!(sensor.F, cloud.F)
    push!(sensor.Nt, size(positions, 2))
    push!(sensor.cand, cand_count)
    push!(sensor.coll, coll_count)
    push!(sensor.n0, n0)
    push!(sensor.Vsim, V)
end

# PLOTTING + ANALYSIS

# Convert a window time to window size
function window_time_size(tseries, window_time)
    return ceil(Int64, length(tseries) * window_time / last(tseries))
end

function temperature_data(sensor, window_size)
    instant_temp = 2 * sensor.ke / (3 * kB)
    rolling_temp = rollmean(instant_temp, window_size)

    final_temp = last(rolling_temp)

    return final_temp, instant_temp, rolling_temp
end

# Average temperature over time
function plot_temperature(sensor, window_time = 1e-2)
    window_size = window_time_size(sensor.time, window_time)
    T_final, instant_temp, rolling_temp = temperature_data(sensor, window_size)
    rolling_time = rollmean(sensor.time, window_size)

    linecolor = RGB(0.1, 0.1, 0.7)

    # Plot instantaneous
    plt = plot(sensor.time, instant_temp,
        title = "Average temperature (Final: $(@sprintf("%.3g", T_final)) K)",
        xlabel = "Time (s)",
        ylabel = "Temperature (K)",
        label = false,
        ylims = (0, Inf),
        linealpha = 0.5,
        linecolor = linecolor,
        dpi = 300)
    # Plot rolling
    plot!(plt, rolling_time, rolling_temp, linecolor = linecolor, label = false)

    return plt, T_final
end

# Total number of atoms over time
function plot_number(sensor)
    Nt = sensor.Nt
    Np = Nt .* sensor.F
    plt = plot(sensor.time, Np,
        title = "Total number of atoms",
        xlabel = "Time (s)",
        ylabel = "Number",
        label = false,
        ylims = (0, Inf),
        dpi = 300)
    return plt
end

# Average kinetic, potential and total energy per atom
function plot_energy(sensor, window_time = 1e-2)
    window_size = window_time_size(sensor.time, window_time)
    rolling_time = rollmean(sensor.time, window_size)

    # Final temp
    T, _, _ = temperature_data(sensor, window_size)

    instant_ke = sensor.ke / (kB * T) # Units kT (final)
    instant_pe = sensor.pe / (kB * T)

    # Rolling and total energy
    rolling_ke, rolling_pe = rollmean.([instant_ke, instant_pe], window_size)
    instant_e, rolling_e = instant_ke + instant_pe, rolling_ke + rolling_pe

    E = last(rolling_e) * kB * T # Final energy (J/atom)

    linecolors = hcat(RGB(0.1059, 0.6196, 0.4667),
                      RGB(0.851, 0.373, 0),
                      RGB(0.459, 0.439, 0.702))

    plt = plot(sensor.time, [instant_ke, instant_pe, instant_e],
        title="Average energy per atom (Final: $(@sprintf("%.3e", E)) J)",
        xlabel="Time (s)",
        ylabel=L"\textrm{Energy\:\:}(k_B T_f)",
        label=false,
        #ylims = (0, Inf),
        linecolor=linecolors,
        linealpha=0.5,
        dpi = 300)
    plot!(rolling_time, [rolling_ke, rolling_pe, rolling_e],
        label=["Kinetic" "Potential" "Total"],
        linecolor=linecolors)

    return plt
end

# Get speeds
function speeds(cloud)
    Nt = cloud.Nt
    velocities = view(cloud.velocities, :, 1:Nt)
    speeds = vec(sqrt.(sum(velocities .* velocities, dims=1)))
    return speeds
end

# Speed distribution
function plot_speed(cloud)
    plt = density(speeds(cloud),
        title="Speed distribution",
        label="Simulation",
        xlabel=L"\textrm{Speed\:\:}(ms^{-1})",
        ylabel="Probability density",
        dpi = 300)

    return plt
end

# Collision rate per atom
function plot_collrate(sensor, window_time = 1e-2)
    window_size = window_time_size(sensor.time, window_time)
    # Collision rates
    timesteps = vcat(first(sensor.time), diff(sensor.time))
    
    colls_per_atom = 2 * sensor.coll ./ sensor.Nt
    instant_collrate = colls_per_atom ./ timesteps

    rolling_time = rollmean(sensor.time, window_size)

    rolling_timesteps = (
        sensor.time[window_size:end] - sensor.time[1:end - window_size + 1]
    )
    rolling_collrate = (
        rolling(sum, colls_per_atom, window_size) ./ rolling_timesteps
    )

    final_rate = last(rolling_collrate)
    linecolor = RGB(0.851, 0.373, 0)
    # Note: per-atom collision rates of test and real particles are equal.
    
    plt = plot(rolling_time, rolling_collrate,
        title="""Average collision rate per atom
        (Final: $(@sprintf("%.3g", final_rate)) Hz)""",
        xlabel="Time (s)",
        ylabel="Rate (Hz)",
        label = false,
        #yaxis = :log10,
        #yminorticks = 10,
        linecolor=linecolor,
        dpi = 300)
    plot!(sensor.time, instant_collrate, label = false,
          linecolor = linecolor, linealpha = 0.5)

    #ylims!(ylims(plt)) # Fix limits so instantaneous rates don't dominate
    plot!(sensor.time, instant_collrate,
        label=false,
        linecolor=linecolor,
        linealpha=0.5)
    
    return plt
end

# Density
function plot_density(sensor, window_time = 1e-2)
    window_size = window_time_size(sensor.time, window_time)
    rolling_time = rollmean(sensor.time, window_size)

    #T, _, _ = temperature_data(sensor, window_size)

    instant_V = sensor.Vsim
    rolling_V = rollmean(instant_V, window_size)

    #V = last(rolling_V)
    instant_n0 = sensor.n0 .* sensor.F
    rolling_n0 = rollmean(instant_n0, window_size)

    rolling_F = rollmean(sensor.F, window_size)
    rolling_Nt = rollmean(sensor.Nt, window_size)

    instant_n_avg = sensor.F .* sensor.Nt ./ instant_V
    rolling_n_avg = rolling_F .* rolling_Nt ./ rolling_V

    linecolors = hcat(RGB(0.1059, 0.6196, 0.4667),
                      RGB(0.851, 0.373, 0))

    plt = plot(sensor.time, [instant_n0, instant_n_avg],
        title="Number density",
        xlabel="Time (s)",
        ylabel=L"n (# $m^{-3}$)",
        label=false,
        linecolor=linecolors,
        linealpha=0.5,
        dpi = 300)
    plot!(rolling_time, [rolling_n0, rolling_n_avg],
        label=["Peak" "Average"],
        linecolor=linecolors)

    return plt
end

# Phase space density (WIP)
function plot_psd(sensor, m, window_time = 1e-2)
    window_size = window_time_size(sensor.time, window_time)
    rolling_time = rollmean(sensor.time, window_size)

    _, _, rolling_T = temperature_data(sensor, window_size)
    rolling_F = rollmean(sensor.F, window_size)
    rolling_n0 = rolling_F .* rollmean(sensor.n0, window_size)
    hbar = 6.626e-34 / (2 * π)
    rolling_psd = rolling_n0 .* (sqrt.(2*π*hbar^2 ./ (m * kB .* rolling_T) ) .^3)

    linecolor = RGB(0.1059, 0.6196, 0.4667)

    plt = plot(rolling_time, rolling_psd,
        title="Phase space density",
        xlabel="Time (s)",
        ylabel=L"PSD",
        label=false,
        linecolor=linecolor,
        dpi = 300)

    return plt
end