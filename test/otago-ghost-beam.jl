using OptEvapCool
using StatsPlots

function ghost_beam_test()
    # Cloud parameters
    Np = 1.6e6
    T₀ = 2e-6
    species = Rb87

    duration = 0.24

    F = Np / (1e4) # TEMPORARILY FIX VALUE OF Nt
    Nc = 5
    Nt = ceil(Int64, Np / F)

    # Beam parameters
    P_h = 0.52 # Horizontal
    P_v = 0.09 # Vertical
    P_g = 0.29 # Ghost

    w_h = 60e-6 # Beam waist (m)
    w_v = 40e-6
    w_g = 40e-6

    dir_h = [0, 0, 1]
    dir_v = [0, -1, 0]
    dir_g = [0, -1, 0]
    
    λ = 1064e-9

    foc_g = (t) -> [85e-6, 0, 0] + (t/0.24) * [-30e-6, 0, 0]

    beam_h = GaussianBeam([0,0,0], dir_h, P_h, w_h, λ)
    beam_v = GaussianBeam([0,0,0], dir_v, P_v, w_v, λ)
    beam_g = GaussianBeam(foc_g, dir_g, P_g, w_g, λ)

    g = 9.81
    gravity = UniformField([0, -g, 0])

    acc1 = acceleration(beam_h)
    acc2 = acceleration(beam_v)
    acc3 = acceleration(gravity)
    acc4 = acceleration(beam_g)

    function accel(p, s, t, o)
        a1 = acc1(p, s, t)
        a2 = acc2(p, s, t)
        a3 = acc3(p, s, t)
        a4 = acc4(p, s, t)
        return (o .= a1 + a2 + a3 + a4)
    end

    pot1 = potential(beam_h)
    pot2 = potential(beam_v)
    pot3 = potential(gravity)
    pot4 = potential(beam_g)

    crossbeam_potential(p, s, t) = pot1(p, s, t) + pot2(p, s, t)

    function total_potential(p, s, t)
        return crossbeam_potential(p, s, t) + pot3(p, s, t) + pot4(p, s, t)
    end

    # Trapping frequencies
    m = species.m
    κ = kappa(species)

    ωr²_v = 8κ*P_v / (m * π * w_v^4) # Radial trapping frequency
    ωa²_v = 4κ*P_v*λ^2 / (m * π^3 * w_v^6) # Axial trapping frequency

    ωr²_h = 8κ*P_h / (m * π * w_h^4)
    ωa²_h = 4κ*P_h*λ^2 / (m * π^3 * w_h^6)

    ωy_un = sqrt(ωr²_h + ωa²_v) # UNPERTURBED
    ωz = sqrt(ωa²_h + ωr²_v)

    # Trap depths
    U_h = 2 * P_h * κ / (π * w_h^2)
    U_v = 2 * P_v * κ / (π * w_v^2)

    # Vertical equilibrium position
    #y0 = - g / (ωy_un^2)

    #println(y0)

    g = 9.81
    q = -4*g^2 / (ωy_un^4 * w_h^2)
    y0 = -w_h/2 * sqrt(- q * exp(q))

    println(y0)

    # Horizontal beam potential at equilibrium position
    U0 = first(pot1(transpose([0 y0 0]), species, 0)) - U_h

    ωxH2 = - 4 / (m * w_h^2) * U0
    ωy = sqrt( - 4 / (m * w_h^2) * U0 * (1 - 4 * y0^2 / (w_h^2)) ) # PERTURBED
    ωx = sqrt(ωxH2 + ωr²_v)

    println(ωx)
    println(ωy)
    println(ωz)

    # Cloud initialisation
    positions = harmonic_boltzmann_positions(Nt, m, T₀, ωx, ωy, ωz)
    velocities = boltzmann_velocities(Nt, m, T₀)

    # Adjust to vertical equilibrium
    centre = [0, y0, 0]
    positions .+= centre

    # Function to make measurements on the system
    sensor = GlobalSensor()
    measure = measurer(sensor, 0.01, ωx, ωy, ωz, centre)

    # Evaporation
    evap = OptEvapCool.ellipsoid_evap(ωx, ωy, ωz, T₀, 1e-3, centre)
    #evap = no_evap
    #evap = energy_evap(1.0 * (U_h + U_v), crossbeam_potential)

    # Three-body loss
    K = 4.3e-29 * 1e-12

    conditions = SimulationConditions(species, F, positions, velocities,
        accel, total_potential, evap = evap, τ_bg = 180, K = K)

    max_dt = 0.05 * 2π / max(ωx, ωy, ωz)
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
    psd_plt = plot_psd(sensor, species.m)

    # Save plots and files
    ft = filetime()
    dir = "./results/$ft-otago-ghost"
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