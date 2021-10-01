# Test of DSMC with Harmonic potential
using Random: MersenneTwister
# using BenchmarkTools
using OptEvapCool
using StatsPlots

function harmonic_none(T₀, Np, duration)
    conditions, sensor, final_cloud, dir = harmonic_test(T₀, Np, duration, name = "no-loss")

    species = conditions.species

     # Final temperature
    window_size = OptEvapCool.window_time_size(sensor.time, 1e-2)
    T_final, _, _ = OptEvapCool.temperature_data(sensor, window_size)
    # Plotting
    temperature_plt = plot_temperature(sensor)

    Nt = final_cloud.Nt
    velocities = view(final_cloud.velocities, :, 1:Nt)
    speeds = vec(sqrt.(sum(velocities .* velocities, dims=1)))
    max_speed = maximum(speeds)

    speed_hist = plot_speed(final_cloud)
    plot!(harmonic_eq_speeds(species.m, T_final, max_speed)...)

    number_plt = plot_number(sensor)
    energy_plt = plot_energy(sensor)
    collrate_plt = plot_collrate(sensor)

    savefig(temperature_plt, "$dir/temp.png")
    savefig(energy_plt, "$dir/energy.png")
    savefig(speed_hist, "$dir/speed.png")
    savefig(collrate_plt, "$dir/collrate.png")
    savefig(number_plt, "$dir/number.png")

    return nothing
end


function harmonic_background(T₀, Np, duration, τ_bg = 5)
    conditions, sensor, final_cloud, dir = harmonic_test(T₀, Np, duration,
        name = "background-loss", τ_bg = τ_bg)

    species = conditions.species

     # Final temperature
    window_size = OptEvapCool.window_time_size(sensor.time, 1e-2)
    T_final, _, _ = OptEvapCool.temperature_data(sensor, window_size)
    # Plotting
    temperature_plt = plot_temperature(sensor)

    Nt = final_cloud.Nt
    velocities = view(final_cloud.velocities, :, 1:Nt)
    speeds = vec(sqrt.(sum(velocities .* velocities, dims=1)))
    max_speed = maximum(speeds)

    speed_hist = plot_speed(final_cloud)
    plot!(harmonic_eq_speeds(species.m, T_final, max_speed)...)

    number_plt = plot_number(sensor)
    plot!(sensor.time, Np * exp.(- sensor.time ./ τ_bg), label = "Theory")

    energy_plt = plot_energy(sensor)
    collrate_plt = plot_collrate(sensor)

    savefig(temperature_plt, "$dir/temp.png")
    savefig(energy_plt, "$dir/energy.png")
    savefig(speed_hist, "$dir/speed.png")
    savefig(collrate_plt, "$dir/collrate.png")
    savefig(number_plt, "$dir/number.png")

    return nothing
end

function harmonic_spontaneous(T₀, Np, duration, η = 10)

    evap = energy_evap(η * kB * T₀)

    conditions, sensor, final_cloud, dir = harmonic_test(T₀, Np, duration,
        name = "spontaneous-evap", evap = evap)

    species = conditions.species

     # Final temperature
    window_size = OptEvapCool.window_time_size(sensor.time, 1e-2)
    T_final, _, _ = OptEvapCool.temperature_data(sensor, window_size)
    # Plotting
    temperature_plt = plot_temperature(sensor)

    Nt = final_cloud.Nt
    velocities = view(final_cloud.velocities, :, 1:Nt)
    speeds = vec(sqrt.(sum(velocities .* velocities, dims=1)))
    max_speed = maximum(speeds)

    speed_hist = plot_speed(final_cloud)
    plot!(harmonic_eq_speeds(species.m, T_final, max_speed)...)

    number_plt = plot_number(sensor)

    energy_plt = plot_energy(sensor)
    collrate_plt = plot_collrate(sensor)

    savefig(temperature_plt, "$dir/temp.png")
    savefig(energy_plt, "$dir/energy.png")
    savefig(speed_hist, "$dir/speed.png")
    savefig(collrate_plt, "$dir/collrate.png")
    savefig(number_plt, "$dir/number.png")

    return nothing
end

function harmonic_forced(T₀, Np, duration, η = 10)

    depth = exponential_ramp(start, stop, tau)
    
    evap = energy_evap(η * kB * T₀)

    conditions, sensor, final_cloud, dir = harmonic_test(T₀, Np, duration,
        name = "spontaneous-evap", evap = evap)

    species = conditions.species
    
     # Final temperature
    window_size = OptEvapCool.window_time_size(sensor.time, 1e-2)
    T_final, _, _ = OptEvapCool.temperature_data(sensor, window_size)
    # Plotting
    temperature_plt = plot_temperature(sensor)

    Nt = final_cloud.Nt
    velocities = view(final_cloud.velocities, :, 1:Nt)
    speeds = vec(sqrt.(sum(velocities .* velocities, dims=1)))
    max_speed = maximum(speeds)

    speed_hist = plot_speed(final_cloud)
    plot!(harmonic_eq_speeds(species.m, T_final, max_speed)...)

    number_plt = plot_number(sensor)

    energy_plt = plot_energy(sensor)
    collrate_plt = plot_collrate(sensor)

    savefig(temperature_plt, "$dir/temp.png")
    savefig(energy_plt, "$dir/energy.png")
    savefig(speed_hist, "$dir/speed.png")
    savefig(collrate_plt, "$dir/collrate.png")
    savefig(number_plt, "$dir/number.png")

    return nothing
end

function harmonic_test(T₀, Np, duration;
        τ_bg = Inf, K = 0, evap = no_evap, name = "test")

    F = 10; Nc = 1;

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

    conditions = SimulationConditions(species, F, positions, velocities,
        acceleration(field), potential(field))

    max_dt = 0.05 * 2π / max(ωx, ωy, ωz)
    # Run evolution
    final_cloud = evolve(conditions, duration;
        Nc = Nc, max_dt = max_dt, measure = measure)

    # Save plots and files
    ft = filetime()
    dir = "./results/$name-$ft"
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

    # display(plt) to display; but this is slow in VSCode.

    return conditions, sensor, final_cloud, dir
end