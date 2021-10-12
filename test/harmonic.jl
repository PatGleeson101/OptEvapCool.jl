# Test of DSMC with Harmonic potential
using Random: MersenneTwister
# using BenchmarkTools
using OptEvapCool
using StatsPlots

# Assumes constant trapping frequencies

function harmonic_plots(conditions, sensor, final_cloud, dir, T₀, depth = 10000)
    species = conditions.species

    # THEORY
    ωx, ωy, ωz = 2π * 150,   2π * 150,   2π * 15

    depth = OptEvapCool.time_parametrize(depth)
    duration = last(sensor.time)
    τ_bg = 100#conditions.τ_bg
    N₀ = size(conditions.positions, 2) * conditions.F

    t_series, N_series, T_series, Γ_series = harmonic_theory(
        species, ωx, ωy, ωz, T₀, N₀, depth, duration, τ_bg
    )

    # Plotting
    temperature_plt, T_final = plot_temperature(sensor)
    plot!(t_series, T_series, label = "Theory")

    speed_hist = plot_speed(final_cloud)
    max_speed = maximum(OptEvapCool.speeds(final_cloud))
    plot!(equilibrium_speeds(species.m, T_final, max_speed)...,
        label = "Theory")

    number_plt = plot_number(sensor)
    plot!(t_series, N_series, label = "Theory")
    energy_plt = plot_energy(sensor)
    # TODO - energy theory
    collrate_plt = plot_collrate(sensor)
    plot!(t_series, Γ_series, label = "Theory")

    savefig(temperature_plt, "$dir/temp.png")
    savefig(energy_plt, "$dir/energy.png")
    savefig(speed_hist, "$dir/speed.png")
    savefig(collrate_plt, "$dir/collrate.png")
    savefig(number_plt, "$dir/number.png")
end

function harmonic_none(T₀, Np, duration)
    results = harmonic_test(T₀, Np, duration, name = "harmonic-no-loss")
    harmonic_plots(results..., T₀)
    return nothing
end

function harmonic_background(T₀, Np, duration, τ_bg = 5)
    results = harmonic_test(T₀, Np, duration,
        name = "harmonic-bg-loss", τ_bg = τ_bg)
    harmonic_plots(results..., T₀)
    return nothing
end

function harmonic_spontaneous(T₀, Np, duration, η = 10)
    depth = η * kB * T₀
    results = harmonic_test(T₀, Np, duration,
        name = "harmonic-spontaneous", depth = depth)

    harmonic_plots(results..., T₀, depth)
    return nothing
end

function harmonic_forced(T₀, Np, duration, τ)
    depth = exponential_ramp(start, stop, τ)
    results = harmonic_test(T₀, Np, duration,
        name = "harmonic-forced", depth = depth, τ_bg = 100)
    harmonic_plots(results..., T₀, depth)
    return nothing
end

function harmonic_test(T₀, Np, duration;
        τ_bg = Inf, K = 0, depth = 10000 #= ≈Inf =#, name = "harmonic")

    F = 10; Nc = 1;

    # TEMPORARY
    F = Np / 10000

    ωx, ωy, ωz = 2π * 150,   2π * 150,   2π * 15
    field = HarmonicField(ωx, ωy, ωz)

    Nt = ceil(Int64, Np / F)

    species = Rb87
    m = Rb87.m

    positions = harmonic_boltzmann_positions(Nt, m, T₀, ωx, ωy, ωz)
    velocities = boltzmann_velocities(Nt, m, T₀)

    # Function to make measurements on the system
    sensor = GlobalSensor()
    measure = measurer(sensor)

    poten = potential(field)
    evap = energy_evap(depth, poten)

    conditions = SimulationConditions(species, F, positions, velocities,
        acceleration(field), poten, K = K, evap = evap, τ_bg = τ_bg)

    max_dt = 0.05 * 2π / max(ωx, ωy, ωz)
    # Run evolution
    final_cloud = evolve(conditions, duration;
        Nc = Nc, max_dt = max_dt, measure = measure)

    # Save files
    ft = filetime()
    dir = "./results/$ft-$name"
    mkpath(dir)

    # Save data to CSV file
    savecsv(sensor, "$dir/sensor-data.csv")

    # Save simulation parameters
    parameter_str = """
    # Harmonic potential test
    Name: $name

    ## Conditions
    Initial temperature: $T₀ K
    Initial number of real particles: $Np
    Duration: $duration
    Background loss time constant: $τ_bg
    Three-body loss rate constant: $K
    Evaporation?: $(evap != no_evap)
    Trapping frequencies:
        $ωx
        $ωy
        $ωz

    ## Numerical parameters
    F = $F
    Nc = $Nc
    Max. timestep: $max_dt
    """

    io = open("$dir/parameters.txt", "w");
    write(io, parameter_str)
    close(io)

    return conditions, sensor, final_cloud, dir
end