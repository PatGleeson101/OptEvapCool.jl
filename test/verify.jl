# Test of DSMC with Harmonic potential
using Random: MersenneTwister
# using BenchmarkTools
using OptEvapCool

function harmonic_test(T₀, Np, duration, τ_bg = 100, K = 1e-29 * 1e-6;
                    F = 1, Nc = 1) # Optional

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

    #=
    # Trap depth
    η = 10 #Truncation parameter
    trap_depth = (t) -> η * kB * T0 #(J)
    #trap_depth = exponential_ramp(, 10
    =#

    conditions = SimulationConditions(species, F, positions, velocities,
        acceleration(field), potential(field))

    max_dt = 0.05 * 2π / max(ωx, ωy, ωz)
    # Run evolution
    final_cloud = evolve(conditions, duration;
        Nc = 1, max_dt = max_dt, measure = measure)

    # Plotting
    temperature_plt = plot_temperature(sensor)
    speed_hist = plot_speed(final_cloud)
    number_plt = plot_number(sensor)
    energy_plt = plot_energy(sensor)
    collrate_plt = plot_collrate(sensor)

    # Save plots and files
    ft = filetime()
    dir = "./results/$ft"
    mkpath(dir)

    savefig(temperature_plt, "$dir/temp_$ft.png")
    savefig(energy_plt, "$dir/energy_$ft.png")
    savefig(speed_hist, "$dir/speed_$ft.png")
    savefig(collrate_plt, "$dir/collrate_$ft.png")
    savefig(number_plt, "$dir/number_$ft.png")

    # Save data to CSV file
    savecsv(sensor, "$dir/sensor-data.csv")

    # display(plt) to display; but this is slow in VSCode.

    return nothing
end