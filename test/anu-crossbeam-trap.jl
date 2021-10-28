using OptEvapCool
using StatsPlots
using Printf: @sprintf
using LaTeXStrings
using Plots.PlotMeasures

function anu_crossbeam_trap(duration = 1.97, input_dir = "")
    # Physical parameters
    Np = 3e7 # Initial atom count
    T₀ = 15e-6 # Initial temperature
    species = Rb87

    # Numerical parameters
    Nt = ceil(Int64, 1e5) # Test particles
    F = Np / Nt
    Nc = 3 # Target number of test particles per cell

    # Beam parameters
    P₁ = exponential_ramp(15, 1.5, 0.8) # Watts
    P₂ = exponential_ramp(7.5, 1.5, 0.8)
    # Set final power to 1.5W to achieve BEC

    w₀ = 130e-6 # Waist (m)
    θ = ( 22.5 * π / 180 ) / 2 # Half-angle between beams

    dir1 = [sin(θ), 0, cos(θ)] # Directions
    dir2 = [-sin(θ), 0, cos(θ)]
    λ₁ = 1064e-9 # Wavelengths
    λ₂ = 1090e-9

    beam1 = GaussianBeam([0,0,0], dir1, P₁, w₀, λ₁)
    beam2 = GaussianBeam([0,0,0], dir2, P₂, w₀, λ₂)

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
    pot_b1! = potential(beam1)
    pot_b2! = potential(beam2)

    function accel_gb(p, s, t, o)
        fill!(o, 0.0)
        acc_b1!(p, s, t, o)
        acc_b2!(p, s, t, o)
        acc_grav!(p, s, t, o)
        return nothing
    end

    function poten_gb(p, s, t, o)
        fill!(o, 0.0)
        pot_b1!(p, s, t, o)
        pot_b2!(p, s, t, o)
        pot_grav!(p, s, t, o)
        return nothing
    end

    sensor_gb = GlobalSensor()
    measure_gb = measurer(sensor_gb, 0.001, ωx, ωy, ωz, centre)

    T_gb(t) = (length(sensor_gb.ke) > 0) ? 2 * last(sensor_gb.ke) / (3 * kB) : T₀
    evap_gb = ellipsoid_evap(ωx, ωy, ωz, T_gb, 1e-6, centre)

    gb_conditions = SimulationConditions(species, F, positions, velocities,
        accel_gb, poten_gb, evap = evap_gb, τ_bg = τ_bg, K = K)

    if input_dir == "" # Simulate new results
        @info "Gaussian beam simulation"
        final_gb_cloud = evolve(gb_conditions, duration;
            Nc = Nc, max_dt = max_dt, measure = measure_gb)
    end

    # HARMONIC SIMULATION
    harm = HarmonicField(ωx, ωy, ωz, centre)
    acc_harm! = acceleration(harm)
    pot_harm! = potential(harm)

    function accel_harm(p, s, t, o)
        fill!(o, 0.0)
        acc_harm!(p, s, t, o)
        acc_grav!(p, s, t, o)
        return nothing
    end

    function poten_harm(p, s, t, o)
        fill!(o, 0.0)
        pot_harm!(p, s, t, o)
        pot_grav!(p, s, t, o)
        return nothing
    end

    function pot_evap_harm(p, s, t, o)
        fill!(o, 0.0)
        pot_harm!(p, s, t, o)
        return nothing
    end

    evap_harm = energy_evap(Uₜ, pot_evap_harm)

    sensor_harm = GlobalSensor()
    measure_harm = measurer(sensor_harm, 0.001, ωx, ωy, ωz, centre)

    harm_conditions = SimulationConditions(species, F, positions, velocities,
        accel_harm, poten_harm, evap = evap_harm, τ_bg = τ_bg, K = K)

    if input_dir == ""
        @info "Harmonic approximation simulation"
        final_harm_cloud = evolve(harm_conditions, duration;
            Nc = Nc, max_dt = max_dt, measure = measure_harm)
    end

    if input_dir != "" # Load results
        sensor_gb = loadsensor("$input_dir/sensor-gb-data.csv")
        sensor_harm = loadsensor("$input_dir/sensor-harm-data.csv")
    end

    # Save sensor data (before plotting, in case plotting fails)
    ft = filetime()
    dir = "./results/$ft-anu-crossbeam-trap"
    mkpath(dir)

    savecsv(sensor_gb, "$dir/sensor-gb-data.csv")
    savecsv(sensor_harm, "$dir/sensor-harm-data.csv")

    # THEORY
    # Pethick model
    peth_t, peth_N, peth_T, peth_Γ = harmonic_theory(
        species, ωx, ωy, ωz, T₀, Np, Uₜ, duration, τ_bg
    )
    # Purdue model
    pur_t, pur_N, pur_T, pur_Γ = purdue_theory(
        species, ωx, ωy, ωz, T₀, Np, Uₜ, duration, τ_bg, K
    )

    # PLOTTING
    default(fontfamily="Computer Modern",
        linewidth=2, framestyle=:box, label=nothing, grid=false)
    scalefontsizes()
    scalefontsizes(1.5)

    window_time = min(1e-2, duration/2)
    gb_time = sensor_gb.time
    harm_time = sensor_harm.time
    gb_window_size = OptEvapCool.window_time_size(gb_time, window_time)
    harm_window_size = OptEvapCool.window_time_size(harm_time, window_time)
    rolling_gb_time = rollmean(gb_time, gb_window_size)
    rolling_harm_time = rollmean(harm_time, harm_window_size)

    # Colour palette
    col1 = RGB(0.0588, 0, 0.5882)
    col2 = RGB(0.9000, 0, 0.3490)
    col3 = RGB(1, 0.6510, 0)
    col4 = RGB(0.0824, 0.8392, 0)

    # Temperature
    rolling_gb_temp = 2 / (3 * kB) * rollmean(sensor_gb.ke, gb_window_size)
    rolling_harm_temp = 2 / (3 * kB) * rollmean(sensor_harm.ke, harm_window_size)

    temp_order = -6
    temp_yformatter(y) = @sprintf("%.1f",y/(10.0^temp_order))
    
    temperature_plt = plot(rolling_gb_time, rolling_gb_temp,
        xlabel = "Time (s)",
        ylabel = L"\textrm{Temperature\ \ }({}\times10^{%$temp_order}\mathrm{K})",
        label = false,
        linecolor = col1,
        ls = :solid,
        ylims = (0, Inf),
        minorticks = 5,
        yformatter = temp_yformatter,
        dpi = 300)
    plot!(rolling_harm_time, rolling_harm_temp,
          label = false, linecolor = col2, ls= :dash)
    plot!(peth_t, peth_T, label = false, linecolor = col3, ls = :dot)
    plot!(pur_t, pur_T, label = false, linecolor = col4, ls = :dashdot)
    
    # Number
    gb_Np = sensor_gb.Nt .* sensor_gb.F
    harm_Np = sensor_harm.Nt .* sensor_harm.F

    num_order = 6
    num_yformatter(y) = @sprintf("%.1f",y/(10.0^num_order))

    number_plt = plot(gb_time, gb_Np,
        xlabel = "Time (s)",
        ylabel = L"\textrm{Number\ \ }({}\times10^{%$num_order})",
        label = false,
        linecolor = col1,
        ls = :solid,
        ylims = (0, Inf),
        minorticks = 5,
        yformatter = num_yformatter,
        dpi = 300)
    plot!(harm_time, harm_Np,
          label = false, linecolor = col2, ls= :dash)
    plot!(peth_t, peth_N, label = false, linecolor = col3, ls = :dot)
    plot!(pur_t, pur_N, label = false, linecolor = col4, ls = :dashdot)

    # Collrate
    rolling_gb_timesteps = (
        gb_time[gb_window_size:end] - gb_time[1:end - gb_window_size + 1]
    )
    rolling_harm_timesteps = (
        harm_time[harm_window_size:end] - harm_time[1:end - harm_window_size + 1]
    )
    rolling_gb_Nt = rollmean(sensor_gb.Nt, gb_window_size)
    rolling_harm_Nt = rollmean(sensor_harm.Nt, harm_window_size)
    rolling_gb_Γ = (2 * rolling(sum, sensor_gb.coll, gb_window_size) ./ 
        (rolling_gb_Nt .* rolling_gb_timesteps)
    )
    rolling_harm_Γ = (2 * rolling(sum, sensor_harm.coll, harm_window_size) ./ 
        (rolling_harm_Nt .* rolling_harm_timesteps)
    )

    collrate_plt = plot(rolling_gb_time, rolling_gb_Γ,
        xlabel = "Time (s)",
        ylabel = "Collision rate (Hz)",
        label = false,
        linecolor = col1,
        ls = :solid,
        dpi = 300)
    plot!(rolling_harm_time, rolling_harm_Γ,
          label = false, linecolor = col2, ls= :dash)
    plot!(peth_t, peth_Γ, label = false, linecolor = col3, ls = :dot)
    plot!(pur_t, pur_Γ, label = false, linecolor = col4, ls = :dashdot)

    # Phase space density
    rolling_gb_n0 = rollmean(sensor_gb.n0 .* sensor_gb.F, gb_window_size)
    rolling_harm_n0 = rollmean(sensor_harm.n0 .* sensor_harm.F, harm_window_size)

    rolling_gb_psd = rolling_gb_n0 .* (
        (2*π*h̄^2 ./ (m * kB .* rolling_gb_temp)) .^1.5
    )
    rolling_harm_psd = rolling_harm_n0 .* (
        (2*π*h̄^2 ./ (m * kB .* rolling_harm_temp)) .^1.5
    )

    psd_plt = plot(rolling_gb_time, rolling_gb_psd,
        xlabel = "Time (s)",
        ylabel = "Phase space density",
        label = false,
        linecolor = col1,
        ls = :solid,
        ylims = (0, Inf),
        minorticks = 5,
        dpi = 300)
    plot!(rolling_harm_time, rolling_harm_psd,
          label = false, linecolor = col2, ls= :dash)
    
    # Number vs temp.
    rolling_gb_Np = rollmean(gb_Np, gb_window_size)
    rolling_harm_Np = rollmean(harm_Np, harm_window_size)

    numtemp_plt = plot(rolling_gb_temp, rolling_gb_Np,
        xlabel = L"\textrm{Temperature\ }(\mu K)",
        ylabel = L"\textrm{Number\ \ }({}\times10^{%$num_order})",
        yformatter = num_yformatter,
        xformatter = temp_yformatter,
        label = false,
        linecolor = col1,
        ls = :solid,
        ylims = (0, Inf),
        xlims = (0, Inf),
        minorticks = 5,
        right_margin = 5mm,
        dpi = 300)
    #=
    plot!(rolling_harm_temp, rolling_harm_Np,
          label = false, linecolor = col2, ls= :dash)
    plot!(peth_T, peth_N, label = false, linecolor = col3, ls = :dot)
    plot!(pur_T, pur_N, label = false, linecolor = col4, ls = :dashdot)
    =#
    # Experimental Results
    scatter!([700e-9], [6e6], color = "black", label = false,
        markersize = 8, markershape = :utriangle)
    #vline!([2.6], line = (:black, 5))

    # Number vs PSD.
    numpsd_plt = plot(rolling_gb_psd, rolling_gb_Np,
        xlabel = "Phase space density",
        ylabel = L"\textrm{Number\ \ }({}\times10^{%$num_order})",
        yformatter = num_yformatter,
        label = false,
        linecolor = col1,
        ls = :solid,
        ylims = (0, Inf),
        xlims = (0, Inf),
        minorticks = 5,
        right_margin = 5mm,
        dpi = 300)
    # Experimental Results
    expt_T = 700e-9
    expt_N = 6e6
    expt_n0 = expt_N * ωx(1.97) * ωy(1.97) * ωz(1.97) * (m / (2π*kB*expt_T))^1.5
    expt_λdB =sqrt(2*π*h̄^2 / (m * kB * expt_T))
    expt_psd = expt_n0 * expt_λdB^3
    scatter!([expt_psd], [expt_N], color = "black", label = false,
        markersize = 8, markershape = :utriangle)
    #vline!([2.6], line = (:black, 5))

    # Save plots
    savefig(temperature_plt, "$dir/temp.png")
    savefig(collrate_plt, "$dir/collrate.png")
    savefig(number_plt, "$dir/number.png")
    savefig(psd_plt, "$dir/psd.png")
    savefig(numtemp_plt, "$dir/numtemp.png")
    savefig(numpsd_plt, "$dir/numpsd.png")

    return nothing
end