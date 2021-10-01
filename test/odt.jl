using OptEvapCool
using StatsPlots

function anu_crossbeam_test()
    F = 1000; Nc = 4;

    P1 = exponential_ramp(15, 2, 0.8) # Watts
    P2 = exponential_ramp(7.5, 2, 0.8)
    # Changing to final power 1.5 would give BEC

    waist = 130e-6 # Waist (m)

    θ = ( 22.5 * π / 180 ) / 2

    Np = 3e7
    T₀ = 15e-6

    duration = 0.01#1.97

    dir1 = [0, 0, 1]
    dir2 = [cos(2*θ), 0, sin(2*θ)]
    λ₁ = 1064e-9
    λ₂ = 1090e-9

    beam1 = GaussianBeam([0,0,0], dir1, P1, waist, λ₁)
    beam2 = GaussianBeam([0,0,0], dir2, P2, waist, λ₂)

    acc1 = acceleration(gravity)
    acc2 = acceleration(beam1)
    acc3 = acceleration(beam2)

    function accel(p, s, t, o)
        a1 = acc1(p, s, t)
        a2 = acc2(p, s, t)
        a3 = acc3(p, s, t)
        return (o .= a1 + a2 + a3)
    end

    pot1 = potential(gravity)
    pot2 = potential(beam1)
    pot3 = potential(beam2)

    function poten(p, s, t)
        p1 = pot1(p, s, t)
        p2 = pot2(p, s, t)
        p3 = pot3(p, s, t)
        return (p1 + p2 + p3)
    end

    #= Expect (with final power 2W)
        Final N: 6e6
        Final T: 700nK - 1 uK
    =#

    Nt = ceil(Int64, Np / F)

    species = Rb87
    m = Rb87.m

    # Approximate initial trapping frequencies
    κ = kappa(species)

    trap_depth(t) = 2 * P1(0) * κ / (π * waist^2)

    U₀ = trap_depth(0)

    zr = OptEvapCool.rayleigh(waist, (λ₁ + λ₂)/2 )
    cosθ2, sinθ2 = cos(θ)^2, sin(θ)^2
    ωz = sqrt(4 * U₀ / m) * sqrt(cosθ2/zr^2 + 2 * sinθ2/ waist^2)
    ωx = sqrt(4 * U₀ / m) * sqrt(sinθ2/zr^2 + 2 * cosθ2/ waist^2)
    ωy = sqrt(8 * U₀ / (m * waist^2))

    # Cloud initialisation
    positions = harmonic_boltzmann_positions(Nt, m, T₀, ωx, ωy, ωz)
    velocities = boltzmann_velocities(Nt, m, T₀)

    # Function to make measurements on the system
    sensor = GlobalSensor()
    measure = measurer(sensor)

    # Evaporation
    evap = energy_evap(trap_depth)

    conditions = SimulationConditions(species, F, positions, velocities,
        accel, poten, evap = evap)

    max_dt = 0.05 * 2π / max(ωx, ωy, ωz)
    # Run evolution
    final_cloud = evolve(conditions, duration;
        Nc = Nc, max_dt = max_dt, measure = measure)

    # Save plots and files
    ft = filetime()
    dir = "./results/crossbeam-$ft"
    mkpath(dir)


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


    # Save data to CSV file
    savecsv(sensor, "$dir/sensor-data.csv")

    # display(plt) to display; but this is slow in VSCode.

    return conditions, sensor, final_cloud, dir
end