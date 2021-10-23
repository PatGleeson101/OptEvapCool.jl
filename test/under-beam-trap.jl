using OptEvapCool
using StatsPlots
using Printf: @sprintf
using LaTeXStrings

function under_beam_trap(duration = 1.97, input_dir = "")
    # Physical parameters
    Np = 3e7 # Initial atom count
    T₀ = 15e-6 # Initial temperature
    species = Rb87

    # Numerical parameters
    Nt = ceil(Int64, 1e4) # Test particles
    F = Np / Nt
    Nc = 3 # Target number of test particles per cell

    #=
    Record 1:
    P1 = 20, P2 = 10, P_g = (t < 1) ? 15 : 15 + 15*(t-1)/0.5,
    foc_g(t) = (t < 1) ? [0, -0.00035 + t*0.00019, 0] : [0, -0.00016, 0]

    Record 2:

    =#

    # Beam parameters
    P₁(t) = (t < 0.4) ? 7.5 : 5
    P₂(t) = (t < 0.4) ? 15 : 10
    P_g(t) = (t < 0.5) ? 15 : 15 + 15*(t-0.5)/0.5

    w₀ = 130e-6 # Waist (m)
    wg = 100e-6
    θ = ( 22.5 * π / 180 ) / 2 # Half-angle between beams
    ϕ = 5 * π / 180 # Tilt angle of under-beam

    dir1 = [sin(θ), 0, cos(θ)] # Directions
    dir2 = [-sin(θ), 0, cos(θ)]
    dir_g = [0, sin(ϕ), cos(ϕ)]

    λ₁ = 1064e-9 # Wavelengths
    λ₂ = 1090e-9
    λ_g = 1090e-9

    #foc_g(t) = exp(-t/0.4)*[0, -0.00041, 0] + (1-exp(-t/0.4))*[0, -0.00016, 0]
    #foc_g(t) = [0, -0.000175-0.00026*0.7*exp(-t/0.6), 0]
    #foc_g(t) = [0, -0.000175-0.00026*0.7*exp(-t/0.3), 0]
    #foc_g(t) = (t < 1) ? [0, -0.00040 + t*0.00023, 0] : [0, -0.00017, 0]
    foc_g(t) = (t < 0.5) ? [0, -0.00040 + t*0.00022/0.5, 0] : [0, -0.00018, 0]

    beam1 = GaussianBeam([0,0,0], dir1, P₁, w₀, λ₁)
    beam2 = GaussianBeam([0,0,0], dir2, P₂, w₀, λ₂)
    beam_g = GaussianBeam(foc_g, dir_g, P_g, wg, λ_g)

    # Trapping frequencies
    m = species.m
    κ = kappa(species)

    Uₜ_coeff = 2 * κ / (π * w₀^2)
    Uₜ(t) = Uₜ_coeff * (P₁(t) + P₂(t)) #Trap depth (without ghost beam)

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
    function y0(t)
        q = -4*g^2 / (ωy(t)^4 * w₀^2)
        return - w₀/2 * sqrt(- q * exp(q))
    end

    centre(t) = [0, y0(t), 0]
    positions .+= centre(0)

    # Three-body loss & background loss
    K = 4.3e-29 * 1e-12
    τ_bg = 180

    # Maximum timestep
    max_dt(t) = 0.05 * 2π / max(ωx(t), ωy(t), ωz(t))

    # Gravitaty
    acc_grav! = acceleration(gravity)
    pot_grav! = potential(gravity)

    # GAUSSIAN BEAM SIMULATION
    acc_b1! = acceleration(beam1)
    acc_b2! = acceleration(beam2)
    acc_g! = acceleration(beam_g)
    pot_b1! = potential(beam1)
    pot_b2! = potential(beam2)
    pot_g! = potential(beam_g)

    function accel(p, s, t, o)
        fill!(o, 0.0)
        acc_b1!(p, s, t, o)
        acc_b2!(p, s, t, o)
        acc_g!(p, s, t, o)
        acc_grav!(p, s, t, o)
        return nothing
    end

    function poten(p, s, t, o)
        fill!(o, 0.0)
        pot_b1!(p, s, t, o)
        pot_b2!(p, s, t, o)
        pot_g!(p, s, t, o)
        pot_grav!(p, s, t, o)
        return nothing
    end

    sensor = GlobalSensor()
    measure = measurer(sensor, 0.0001, ωx, ωy, ωz, centre)

    T(t) = (length(sensor.ke) > 0) ? 2 * last(sensor.ke) / (3 * kB) : T₀
    evap = ellipsoid_evap(ωx, ωy, ωz, T, 1e-6, centre)

    conditions = SimulationConditions(species, F, positions, velocities,
        accel, poten, evap = evap, τ_bg = τ_bg, K = K)

    if input_dir == "" # Simulate new results
        final_cloud = evolve(conditions, duration;
            Nc = Nc, max_dt = max_dt, measure = measure)
    else # Load provided results
        sensor = loadsensor("$input_dir/sensor-data.csv")
    end

    # Save sensor data before plotting
    ft = filetime()
    dir = "./results/$ft-under-beam-trap"
    mkpath(dir)
    savecsv(sensor, "$dir/sensor-data.csv")

    # PLOTTING
    default(fontfamily="Computer Modern",
        linewidth=2, framestyle=:box, label=nothing, grid=false)
    #scalefontsizes(1.3)

    window_time = min(duration/2, 4e-2)
    time = sensor.time
    window_size = OptEvapCool.window_time_size(time, window_time)
    rolling_time = rollmean(time, window_size)

    # Colour palette
    col1 = RGB(0.0588, 0, 0.5882)
    #col2 = RGB(0.9000, 0, 0.3490)
    #col3 = RGB(1, 0.6510, 0)
    #col4 = RGB(0.0824, 0.8392, 0)

    # Temperature
    rolling_temp = 2 / (3 * kB) * rollmean(sensor.ke, window_size)

    temp_order = -6
    temp_yformatter(y) = @sprintf("%.2f",y/(10.0^temp_order))
    
    temperature_plt = plot(rolling_time, rolling_temp,
        xlabel = "Time (s)",
        ylabel = L"\textrm{Temperature\ \ }({}\times10^{%$temp_order}\mathrm{K})",
        label = false,
        linecolor = col1,
        ls = :solid,
        #yscale = :log10,
        yminorticks = 10,
        yformatter = temp_yformatter,
        ylims = (0, Inf),
        dpi = 300)
    
    # Number
    Np = sensor.Nt .* sensor.F

    num_order = 6
    num_yformatter(y) = @sprintf("%.2f",y/(10.0^num_order))

    number_plt = plot(time, Np,
        xlabel = "Time (s)",
        ylabel = L"\textrm{Number\ \ }({}\times10^{%$num_order})",
        label = false,
        linecolor = col1,
        ls = :solid,
        #yscale = :log10,
        yminorticks=10,
        yformatter = num_yformatter,
        ylims = (0, Inf),
        dpi = 300)

    # Collrate
    rolling_timesteps = (
        time[window_size:end] - time[1:end - window_size + 1]
    )
    rolling_Nt = rollmean(sensor.Nt, window_size)
    rolling_Γ = (2 * rolling(sum, sensor.coll, window_size) ./ 
        (rolling_Nt .* rolling_timesteps)
    )

    collrate_plt = plot(rolling_time, rolling_Γ,
        xlabel = "Time (s)",
        ylabel = "Collision rate (Hz)",
        label = false,
        linecolor = col1,
        ls = :solid,
        dpi = 300)

    # Phase space density
    rolling_n0 = rollmean(sensor.n0 .* sensor.F, window_size)
    rolling_psd = rolling_n0 .* (
        (2*π*h̄^2 ./ (m * kB .* rolling_temp)) .^1.5
    )

    psd_plt = plot(rolling_time, rolling_psd,
        xlabel = "Time (s)",
        ylabel = "Phase space density",
        label = false,
        linecolor = col1,
        ylims = (0, Inf),
        ls = :solid,
        dpi = 300)

    # Save plots
    savefig(temperature_plt, "$dir/temp.png")
    savefig(collrate_plt, "$dir/collrate.png")
    savefig(number_plt, "$dir/number.png")
    savefig(psd_plt, "$dir/psd.png")

    return nothing
end