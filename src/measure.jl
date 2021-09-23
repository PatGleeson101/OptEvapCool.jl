# Performing and plotting measurements during the simulation
using Statistics: mean
using RollingFunctions: rollmean, rolling
using Printf: @sprintf
using LaTeXStrings
using StatsPlots: plot, plot!, density
# using StatsBase: fit, Histogram
using Dates: DateFormat, format, now
using DataFrames: DataFrame
using CSV

# Dummy function
null(args...) = nothing

# Compute translational kinetic energy
function avg_kinetic_energy(velocities, m)
    # The extra 3 is to account for the fact that the velocities
    # array has 3xN elements, not just N.
    return 0.5 * m * 3 * mean(velocities .^2)
end

# Get a time-save string
filetime_fmt = DateFormat("y-m-d-H-M-S")
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

    GlobalSensor() = new([zeros(0) for _ in 1:7]...)
end

# TODO: GridSensor

# Overload measurement
measurer(sensor::GlobalSensor) = (args...) -> measure(sensor, args...)

# Save to CSV
function savecsv(sensor::GlobalSensor, filename)
    CSV.write(filename, DataFrame(
        time = sensor.time,
        ke = sensor.ke,
        pe = sensor.pe,
        F = sensor.F,
        Nt = sensor.Nt,
        cand = sensor.cand,
        coll = sensor.coll
    ))
    return nothing
end

# Perform measurements
function measure(sensor::GlobalSensor, cloud, conditions, cand_count, coll_count, t)
    Nt = cloud.Nt
    species = conditions.species
    positions = view(cloud.positions, :, Nt)
    velocities = view(cloud.velocities, :, Nt)

    ke = avg_kinetic_energy(velocities, species.m)
    pe = mean( conditions.potential(positions, species, t) )

    push!(sensor.time, t)
    push!(sensor.ke, ke)
    push!(sensor.pe, pe)
    push!(sensor.F, cloud.F)
    push!(sensor.Nt, Nt)
    push!(sensor.cand, cand_count)
    push!(sensor.coll, coll_count)
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

function plot_temperature(sensor, window_time = 1e-2)
    window_size = window_time_size(sensor.time, window_time)
    T, instant_temp, rolling_temp = temperature_data(sensor, window_size)
    rolling_time = rollmean(sensor.time, window_size)

    linecolor = RGB(0.1, 0.1, 0.7)

    # Plot instantaneous
    plt = plot(sensor.time, instant_temp,
        title = "Average temperature (Final: $(@sprintf("%.3g", T)) K)",
        xlabel = "Time (s)",
        ylabel = "Temperature (K)",
        label = false,
        linealpha = 0.5,
        linecolor = linecolor,
        dpi = 300)
    plot!(plt, rolling_time, rolling_temp, linecolor = linecolor, label = false)
    
    # TODO: theory
    # T_theory = E_initial / (3 * kB)
    #hline!([T_theory], linestyle=:dash, label="Theory")

    return plt
end

function plot_number(sensor)
    Nt = sensor.Nt
    Np = Nt .* sensor.F
    plt = plot(sensor.time, [Nt, Np],
        title = "Total number of atoms",
        xlabel = "Time (s)",
        yabel = "Number",
        label = ["Test" "Real"],
        dpi = 300)
    return plt
end

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
        title="Average energy per atom (Final: $(@sprintf("%.3g", E)) J)",
        xlabel="Time (s)",
        ylabel=L"\textrm{Energy\:\:}(k_B T_{final})",
        label=false,
        ylim=(0, 1.2 * maximum(instant_e)),
        linecolor=linecolors,
        linealpha=0.5,
        dpi = 300)
    plot!(rolling_time, [rolling_ke, rolling_pe, rolling_e],
        label=["Kinetic" "Potential" "Total"],
        linecolor=linecolors)
    
    #= TODO: theory
    E_final = last(rolling_e) * kB * T_theory # Final energy per particle in J
    # Theoretical final total energy/atom (units kT)
    E_theory = E_initial / (kB * T_theory)
    hline!([E_theory E_theory / 2],
            linestyle=:dash,
            label=["Total (theory)" "Potential/Kinetic (theory)"])
    =#

    return plt
end

# Speed distribution
function plot_speed(cloud)
    Nt = cloud.Nt
    velocities = view(cloud.velocities, :, Nt)
    ke_squared = sum(velocities .* velocities, dims=1)
    speeds = vec(sqrt.(ke_squared))

    plt = density(speeds,
        title="Speed distribution",
        label="Simulation",
        xlabel=L"\textrm{Speed\:\:}(ms^{-1})",
        ylabel="Probability density",
        dpi = 300)

    # TODO: theory
    #=
    # Ideal velocity probability distribution in thermal equilibrium
    N0 = 4π * (m / (2π * kB * T_final))^1.5; # Normalisation constant ##NOTE SHIFT TO T_FINAL
    fv(v) = N0 * v^2 * exp(-0.5 * m * v^2 / (kB * T_final)) # Ideal distribution
    speed_domain = range(0, stop=1.2 * maximum(final_speeds), length=300)
    theoretical_speed_density = fv.(speed_domain)
    plot!(speed_domain, theoretical_speed_density, label="Theory")
    =#

    return plt
end

function plot_collrate(sensor, window_time = 1e-2)
    window_size = window_time_size(sensor.time, window_time)
    # Collision rates
    timesteps = vcat(first(sensor.time), diff(sensor.time))
    
    colls_per_atom = 2 * sensor.colls ./ sensor.Nt
    instant_collrate = colls_per_atom ./ timesteps
    # TODO: candidate rate

    rolling_timesteps = (
        time_series[window_size:end] - time_series[1:end - window_size + 1]
    )
    rolling_collrate = (
        rolling(sum, colls_per_atom, window_size) ./ rolling_timesteps
    )

    final_rate = last(rolling_collrate)
    linecolor = RGB(0.851, 0.373, 0)
    # Note: per-atom collision rates of test and real particles are equal.
    
    plt = plot(rolling_time, rolling_coll_rate,
        title="""Average collision rate per atom
        (Final: $(@sprintf("%.3g", final_rate)) Hz)""",
        xlabel="Time (s)",
        ylabel="Rate (Hz)",
        label = false,
        yaxis = :log10,
        yminorticks = 10,
        linecolor=linecolor,
        dpi = 300)

    ylims!(ylims(plt)) # Fix limits so instantaneous rates don't dominate
    plot!(sensor.time, instant_collrate,
        label=false,
        linecolor=linecolor,
        linealpha=0.1)

    #= TODO: theory
    # Theoretical mean density and speed (harmonic trap, no loss)
    #hline!([rate_theory], linestyle=:dash, label="Theory (real)")
    nbar = Np * ω_x * ω_y * ω_z * (m / (4π * kB * T_theory))^1.5
    vbar = sqrt(8 * kB * T_theory / (π * m))
    rate_theory = 1 / sqrt(2) * Np * nbar * σ * vbar
    =#
end