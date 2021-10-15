using OptEvapCool
using StatsPlots

function anu_crossbeam_test()
    #= Expect (with final power 2W)
        Final N: 6e6
        Final T: 700nK - 1 uK
        # Changing to final power 1.5 would give BEC
    =#

    # Cloud parameters
    Np = 3e7
    T₀ = 15e-6
    species = Rb87

    duration = 1.97

    F = Np / (1e4) # TEMPORARILY FIX VALUE OF Nt
    Nc = 5
    Nt = ceil(Int64, Np / F)

    # Beam parameters
    P₁ = exponential_ramp(15, 2, 0.8) # Watts
    P₂ = exponential_ramp(7.5, 2, 0.8)

    w₀ = 130e-6 # Beam waist (m)
    θ = ( 22.5 * π / 180 ) / 2 # Half-angle between crossed beams

    dir1 = [sin(θ), 0, cos(θ)]
    dir2 = [-sin(θ), 0, cos(θ)]
    λ₁ = 1064e-9
    λ₂ = 1090e-9

    beam1 = GaussianBeam([0,0,0], dir1, P₁, w₀, λ₁)
    beam2 = GaussianBeam([0,0,0], dir2, P₂, w₀, λ₂)

    acc1 = acceleration(beam1)
    acc2 = acceleration(beam2)
    acc3 = acceleration(gravity)

    function accel(p, s, t, o)
        a1 = acc1(p, s, t)
        a2 = acc2(p, s, t)
        a3 = acc3(p, s, t)
        return (o .= a1 + a2 + a3)
    end

    pot1 = potential(beam1)
    pot2 = potential(beam2)
    pot3 = potential(gravity)

    function crossbeam_potential(p, s, t)
        return pot1(p, s, t) + pot2(p, s, t)
    end

    function total_potential(p, s, t)
        return pot3(p, s, t) + crossbeam_potential(p, s, t)
    end

    # Trapping frequencies
    m = species.m
    κ = kappa(species)

    Uₜ_coeff = 2 * κ / (π * w₀^2)
    Uₜ(t) = Uₜ_coeff * (P₁(t) + P₂(t)) #Trap depth

    ωx_coeff = sqrt(4 * cos(θ)^2 / (m * w₀^2))
    ωz_coeff = sqrt(4 * sin(θ)^2 / (m * w₀^2))
    ωy_coeff = sqrt(4 / (m * w₀^2))

    ωx(t) = ωx_coeff * sqrt(Uₜ(t))
    ωz(t) = ωz_coeff * sqrt(Uₜ(t))
    ωy(t) = ωy_coeff * sqrt(Uₜ(t))

    # Cloud initialisation
    positions = harmonic_boltzmann_positions(Nt, m, T₀, ωx(0), ωy(0), ωz(0))
    velocities = boltzmann_velocities(Nt, m, T₀)

    # Adjust vertical equilibrium position
    g = 9.81
    q = -4*g^2 / (ωy(0)^4 * w₀^2)
    y0 = w₀/2 * sqrt(- q * exp(q))

    positions .+= [0, -y0, 0]

    # Function to make measurements on the system
    sensor = GlobalSensor()
    measure = measurer(sensor, 0.01, ωx, ωy, ωz)

    # Evaporation
    #evap = energy_evap(Uₜ, crossbeam_potential)
    #evap = OptEvapCool.no_evap
    evap = OptEvapCool.ellipsoid_evap(ωx(2), ωy(2), ωz(2), T₀, 1e-4)

    # Three-body loss
    K = 4.3e-29 * 1e-12

    conditions = SimulationConditions(species, F, positions, velocities,
        accel, total_potential, evap = evap, τ_bg = 180, K = K)

    max_dt(t) = 0.05 * 2π / max(ωx(t), ωy(t), ωz(t))
    # Run evolution
    final_cloud = evolve(conditions, duration;
        Nc = Nc, max_dt = max_dt, measure = measure)

    # Plotting
    temperature_plt, T_final = plot_temperature(sensor)

    max_speed = maximum(OptEvapCool.speeds(final_cloud))
    speed_hist = plot_speed(final_cloud)
    plot!(equilibrium_speeds(m, T_final, max_speed)...,
        label = "Theory")

    number_plt = plot_number(sensor)
    energy_plt = plot_energy(sensor)
    collrate_plt = plot_collrate(sensor)
    density_plt = plot_density(sensor)
    psd_plt = plot_psd(sensor, m)

    # Save plots and files
    ft = filetime()
    dir = "./results/$ft-crossbeam"
    mkpath(dir)

    savefig(temperature_plt, "$dir/temp.png")
    savefig(energy_plt, "$dir/energy.png")
    savefig(speed_hist, "$dir/speed.png")
    savefig(collrate_plt, "$dir/collrate.png")
    savefig(number_plt, "$dir/number.png")
    savefig(density_plt, "$dir/density.png")
    savefig(psd_plt, "$dir/psd.png")

    savecsv(sensor, "$dir/sensor-data.csv")

    return nothing
end

function macro_fort()
    # Expect final N: 3e5 @ critical temperature

    # Cloud parameters
    Np = 3e6
    T₀ = 65e-6
    species = Rb87

    duration = 3.4

    F = Np / (1e4) # TEMPORARILY FIX VALUE OF Nt
    Nc = 5
    Nt = ceil(Int64, Np / F)

    # Beam parameters
    # Wide beam
    P1_stage1 = linear_ramp(28, 7, 1.4)
    P1_stage2 = linear_ramp(7, 20, 2)
    P₁(t) = (t < 1.4) ? P1_stage1(t) : P1_stage2(t)

    # Tight beam
    P2_stage1 = exponential_ramp(6, 0, 0.34)
    P₂(t) = (t < 1.4) ? P2_stage1(t) : 0.1

    w₁ = 180e-6 # Beam waist (m)
    w₂ = 26e-6
    θ = 56 * π/180 # FULL angle between crossed beams

    dir1 = [sin(θ), 0, cos(θ)]
    dir2 = [0, 0, 1]
    λ₁ = 1565e-9
    λ₂ = 1565e-9

    wide_beam = GaussianBeam([0,80e-6,0], dir1, P₁, w₁, λ₁)
    tight_beam = GaussianBeam([0,0,0], dir2, P₂, w₂, λ₂)

    acc1 = acceleration(wide_beam)
    acc2 = acceleration(tight_beam)
    acc3 = acceleration(gravity)

    function accel(p, s, t, o)
        a1 = acc1(p, s, t)
        a2 = acc2(p, s, t)
        a3 = acc3(p, s, t)
        return (o .= a1 + a2 + a3)
    end

    pot1 = potential(wide_beam)
    pot2 = potential(tight_beam)
    pot3 = potential(gravity)

    function total_potential(p, s, t)
        return pot3(p, s, t) + pot1(p, s, t) + pot2(p, s, t)
    end

    # Trapping frequencies
    m = species.m

    P(t) = P₂(t)
    P0 = P(0)
    ωx0 = 3800
    ωy0 = 3800
    ωz0 = 124

    ωx(t) = ωx0 * sqrt(P(t)/P0)
    ωz(t) = ωz0 * sqrt(P(t)/P0)
    ωy(t) = ωy0 * sqrt(P(t)/P0)

    # Cloud initialisation
    positions = harmonic_boltzmann_positions(Nt, m, T₀, ωx(0), ωy(0), ωz(0))
    velocities = boltzmann_velocities(Nt, m, T₀)

    # Function to make measurements on the system
    sensor = GlobalSensor()
    measure = measurer(sensor, 0.001, ωx, ωy, ωz)

    # Evaporation
    evap = OptEvapCool.ellipsoid_evap(ωx(0), ωy(0), ωz(0), T₀, 1e-150)

    # Three-body loss
    K = 4.3e-29 * 1e-12

    conditions = SimulationConditions(species, F, positions, velocities,
        accel, total_potential, evap = evap, τ_bg = 180, K = K)

    max_dt = 0.05 * 2π / max(ωx(0), ωy(0), ωz(0))
    # Run evolution
    final_cloud = evolve(conditions, duration;
        Nc = Nc, max_dt = max_dt, measure = measure)

    # Plotting
    temperature_plt, T_final = plot_temperature(sensor)

    max_speed = maximum(OptEvapCool.speeds(final_cloud))
    speed_hist = plot_speed(final_cloud)
    plot!(equilibrium_speeds(m, T_final, max_speed)...,
        label = "Theory")

    number_plt = plot_number(sensor)
    energy_plt = plot_energy(sensor)
    collrate_plt = plot_collrate(sensor)
    density_plt = plot_density(sensor)
    psd_plt = plot_psd(sensor, m)

    # Save plots and files
    ft = filetime()
    dir = "./results/$ft-macro-fort"
    mkpath(dir)

    savefig(temperature_plt, "$dir/temp.png")
    savefig(energy_plt, "$dir/energy.png")
    savefig(speed_hist, "$dir/speed.png")
    savefig(collrate_plt, "$dir/collrate.png")
    savefig(number_plt, "$dir/number.png")
    savefig(density_plt, "$dir/density.png")
    savefig(psd_plt, "$dir/psd.png")

    savecsv(sensor, "$dir/sensor-data.csv")

    return nothing
end


function harmonic_anu_crossbeam()
    #= Expect (with final power 2W)
        Final N: 6e6
        Final T: 700nK - 1 uK
        # Changing to final power 1.5 would give BEC
    =#

    # Cloud parameters
    Np = 3e7
    T₀ = 15e-6
    species = Rb87

    duration = 1.97

    F = Np / (1e4) # TEMPORARILY FIX VALUE OF Nt
    Nc = 5
    Nt = ceil(Int64, Np / F)

    # Beam parameters
    P₁ = exponential_ramp(15, 2, 0.8) # Watts
    P₂ = exponential_ramp(7.5, 2, 0.8)

    w₀ = 130e-6 # Beam waist (m)
    θ = ( 22.5 * π / 180 ) / 2 # Half-angle between crossed beams

    # Trapping frequencies
    m = species.m
    κ = kappa(species)

    Uₜ_coeff = 2 * κ / (π * w₀^2)
    Uₜ(t) = Uₜ_coeff * (P₁(t) + P₂(t)) #Trap depth

    ωx_coeff = sqrt(4 * cos(θ)^2 / (m * w₀^2))
    ωz_coeff = sqrt(4 * sin(θ)^2 / (m * w₀^2))
    ωy_coeff = sqrt(4 / (m * w₀^2))

    ωx(t) = ωx_coeff * sqrt(Uₜ(t))
    ωz(t) = ωz_coeff * sqrt(Uₜ(t))
    ωy(t) = ωy_coeff * sqrt(Uₜ(t))

    field = HarmonicField(ωx, ωy, ωz)

    acc1 = acceleration(field)
    acc2 = acceleration(gravity)

    function accel(p, s, t, o)
        a1 = acc1(p, s, t)
        a2 = acc2(p, s, t)
        return (o .= a1 + a2)
    end

    pot1 = potential(field)
    pot2 = potential(gravity)

    function total_potential(p, s, t)
        return pot1(p, s, t) + pot2(p, s, t)
    end

    # Cloud initialisation
    positions = harmonic_boltzmann_positions(Nt, m, T₀, ωx(0), ωy(0), ωz(0))
    velocities = boltzmann_velocities(Nt, m, T₀)

    # Function to make measurements on the system
    sensor = GlobalSensor()
    measure = measurer(sensor, 0.001, ωx, ωy, ωz)

    # Evaporation
    evap = energy_evap(Uₜ, pot1)
    #evap = OptEvapCool.no_evap
    #evap = OptEvapCool.ellipsoid_evap(ωx(2), ωy(2), ωz(2), T₀, 1e-150)

    # Three-body loss
    K = 1e-29 * 1e-12

    conditions = SimulationConditions(species, F, positions, velocities,
        accel, total_potential, evap = evap, τ_bg = 180, K = K)

    max_dt = 0.05 * 2π / max(ωx(0), ωy(0), ωz(0))
    # Run evolution
    final_cloud = evolve(conditions, duration;
        Nc = Nc, max_dt = max_dt, measure = measure)

    # Plotting
    temperature_plt, T_final = plot_temperature(sensor)

    max_speed = maximum(OptEvapCool.speeds(final_cloud))
    speed_hist = plot_speed(final_cloud)
    plot!(equilibrium_speeds(m, T_final, max_speed)...,
        label = "Theory")

    number_plt = plot_number(sensor)
    energy_plt = plot_energy(sensor)
    collrate_plt = plot_collrate(sensor)

    # Save plots and files
    ft = filetime()
    dir = "./results/$ft-crossbeam-harmonic"
    mkpath(dir)

    savefig(temperature_plt, "$dir/temp.png")
    savefig(energy_plt, "$dir/energy.png")
    savefig(speed_hist, "$dir/speed.png")
    savefig(collrate_plt, "$dir/collrate.png")
    savefig(number_plt, "$dir/number.png")

    savecsv(sensor, "$dir/sensor-data.csv")

    return nothing
end