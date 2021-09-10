# Test of DSMC with Harmonic potential
using Random: MersenneTwister
using Printf: @sprintf
using LaTeXStrings
using RollingFunctions: rollmean, rolling
using StatsPlots # supersedes Plots
# using StatsBase: fit, Histogram
# using BenchmarkTools
using OptEvapCool

# Wrapper method to run with specific seed.
function test(seed)
    test(rng=MersenneTwister(seed))
end

function test(Np, duration, F, Nc, rng=MersenneTwister())
    # Np: initial number
    # duration: vitrual simulation time (s)
    # F: number of real particles per test particle
    # Nc: target average number of atoms per cell

    # Measurement storage
    time_series = zeros(0)
    pe_series = zeros(0)
    ke_series = zeros(0)
    coll_counts = zeros(0) # Collision counts
    cand_counts = zeros(0) # Candidate counts

    # Simulation parameters
    # Physical particles
    m = 1.44e-25 # Atomic mass (kg)
    a_sc = 1e-8 # s-wave scattering length
    σ = 8 * pi * a_sc^2 # Total collision cross section
    # T0 = 1e-6 # Approximate initial temperature (K)
    ω_x = 2π * 150; # Trapping frequencies
    ω_y = 2π * 150;
    ω_z = 2π * 15;
    accel, potential = harmonic_field(ω_x, ω_y, ω_z)

    # Test particles
    Nt = ceil(Int64, Np / F) # Initial number

    # Cube initialisation (at T = approx. 1 microkelvin)
    positions, velocities = uniform_cube_cloud(Nt, 1e-5, 0.01)

    # Initial average energy per particle
    E_initial = (avg_kinetic_energy(velocities, m)
        + avg_potential_energy(positions, potential, m, 0))

    # Function to make measurements on the system
    measure = record(time_series, ke_series, pe_series,
                     cand_counts, coll_counts, m, potential)

    # Run evolution
    final_pos, final_vel = evolve(positions, velocities, accel, duration, σ,
        ω_x, m, F = F, Nc = Nc, measure = measure, rng = rng)
    
    # PLOTTING + ANALYSIS
    # Set up rolling window
    window_time = 1e-2 # Desired window time interval
    # Get window size in number of iterations
    window_size = ceil(Int64, length(time_series) * window_time / duration)
    rolling_time = rollmean(time_series, window_size)

    # Temperature
    instant_temperature = 2 * ke_series / (3 * kB)
    rolling_temperature = rollmean(instant_temperature, window_size)
    # Note: averaging the instantaneous temperature is inaccurate
    # within a *single* cell, but over all cells is fine.
    T_final = last(rolling_temperature)
    T_theory = E_initial / (3 * kB)
    linecolor = RGB(0.1, 0.1, 0.7)

    temperature_plt = plot(time_series, instant_temperature,
        title="Average temperature (Final: $(@sprintf("%.3g", T_final)) K)",
        xlabel="Time (s)",
        ylabel="Temperature (K)",
        label=false,
        linealpha=0.5,
        linecolor=linecolor)
    plot!(rolling_time, rolling_temperature,
          linecolor=linecolor, label="Simulation")
    hline!([T_theory], linestyle=:dash, label="Theory")

    # Energy
    instant_ke = ke_series / (kB * T_theory) # KE in units of kT (final)
    instant_pe = pe_series / (kB * T_theory)
    rolling_ke, rolling_pe = rollmean.([instant_ke, instant_pe], window_size)
    instant_e, rolling_e = instant_ke + instant_pe, rolling_ke + rolling_pe
    E_final = last(rolling_e) * kB * T_theory # Final energy per particle in J
    # Theoretical final total energy/atom (units kT)
    E_theory = E_initial / (kB * T_theory)
    linecolors = hcat(RGB(0.1059, 0.6196, 0.4667),
                      RGB(0.851, 0.373, 0),
                      RGB(0.459, 0.439, 0.702))

    energy_plt = plot(time_series, [instant_ke, instant_pe, instant_e],
        title="Average energy per atom (Final: $(@sprintf("%.3g", E_final)) J)",
        xlabel="Time (s)",
        ylabel=L"\textrm{Energy\:\:}(k_B T_{final})",
        label=false,
        ylim=(0, 1.2 * maximum(instant_e)),
        linecolor=linecolors,
        linealpha=0.5)
    plot!(rolling_time, [rolling_ke, rolling_pe, rolling_e],
        label=["Kinetic" "Potential" "Total"],
        linecolor=linecolors)
    hline!([E_theory E_theory / 2],
            linestyle=:dash,
            label=["Total (theory)" "Potential/Kinetic (theory)"])
    
    # Cloud distribution
    final_speeds = vec(sqrt.(sum(final_vel .* final_vel, dims=1)))
    # Ideal velocity probability distribution in thermal equilibrium
    N0 = 4π * (m / (2π * kB * T_theory))^1.5; # Normalisation constant
    fv(v) = N0 * v^2 * exp(-0.5 * m * v^2 / (kB * T_theory)) # Ideal distribution
    speed_domain = range(0, stop=1.2 * maximum(final_speeds), length=300)
    theoretical_speed_density = fv.(speed_domain)
    # Plot ideal and actual speed distributions
    speed_hist = density(final_speeds,
                         title="Speed distribution",
                         label="Simulation",
                         xlabel=L"\textrm{Speed\:\:}(ms^{-1})",
                         ylabel="Probability density")
    plot!(speed_domain, theoretical_speed_density, label="Theory")

    # TODO: position distribution #fit(Histogram, final_pos[1,:])

    # Collision rates
    timesteps = vcat(first(time_series), diff(time_series))
    instant_coll_rate = coll_counts ./ timesteps
    instant_cand_rate = cand_counts ./ timesteps

    rolling_timesteps = time_series[window_size:end] - time_series[1:end - window_size + 1]
    rolling_coll_rate = rolling(sum, coll_counts, window_size) ./ rolling_timesteps
    rolling_cand_rate = rolling(sum, cand_counts, window_size) ./ rolling_timesteps

    rate_final = F * last(rolling_coll_rate)
    # Theoretical mean density and speed
    nbar = Np * ω_x * ω_y * ω_z * (m / (4π * kB * T_theory))^1.5
    vbar = sqrt(8 * kB * T_theory / (π * m))
    σ = 8π * a_sc^2
    rate_theory = 1 / sqrt(2) * Np * nbar * σ * vbar

    collision_plt = plot(rolling_time,
        [rolling_cand_rate,
         rolling_coll_rate,
         F * rolling_coll_rate],
        label=hcat("Candidates (test)",
                   "Collisions (test)",
                   "Collisions (real)"),
        title="Total collision rate (Final: $(@sprintf("%.3g", rate_final))/s)",
        xlabel="Time (s)",
        ylabel=L"\textrm{Rate\:\:}(s^{-1})",
        yaxis = :log,
        linecolor=linecolors)
    ylims!(ylims(collision_plt)) # Fix limits so instantaneous rates don't dominate
    plot!(time_series,
        [instant_cand_rate,
         instant_coll_rate,
         F * instant_coll_rate],
        label=false,
        linecolor=linecolors,
        linealpha=0.5)
    hline!([rate_theory], linestyle=:dash, label="Theory (real)")

    display(temperature_plt)
    display(energy_plt)
    display(speed_hist)
    display(collision_plt)
    return nothing
end