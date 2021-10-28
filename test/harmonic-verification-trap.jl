using OptEvapCool
using StatsPlots
using Printf: @sprintf
using LaTeXStrings
using Plots.PlotMeasures

function harmonic_verification_trap(duration = 2, input_dir = "")
    # Physical parameters
    Np = 1e7 # Initial atom count
    T₀ = 1e-6 # Initial temperature
    species = Rb87

    # Numerical parameters
    Nt = ceil(Int64, 1e5) # Test particles
    F = Np / Nt
    Nc = 3 # Target number of test particles per cell

    # Trapping frequencies
    m = species.m
    ωx = 2π * 150
    ωy = 2π * 150
    ωz = 2π * 15

    # Cloud initialisation
    positions = harmonic_boltzmann_positions(Nt, m, T₀, ωx, ωy, ωz)
    velocities = boltzmann_velocities(Nt, m, T₀)

    # Three-body loss & background loss
    K = 4.3e-29 * 1e-12
    τ_bg = 180

    # Maximum timestep
    max_dt = 0.05 * 2π / max(ωx, ωy, ωz)

    # HARMONIC SIMULATION
    field = HarmonicField(ωx, ωy, ωz)
    acc! = acceleration(field)
    pot! = potential(field)

    function accel(p, s, t, o)
        fill!(o, 0.0)
        acc!(p, s, t, o)
        return nothing
    end

    function poten(p, s, t, o)
        fill!(o, 0.0)
        pot!(p, s, t, o)
        return nothing
    end

    # Evaporation
    η = 10
    Uₜ = η * kB * T₀
    evap = energy_evap(Uₜ, poten)

    sensor = GlobalSensor()
    measure = measurer(sensor, 0.001, ωx, ωy, ωz)

    conditions = SimulationConditions(species, F, positions, velocities,
        accel, poten, evap = evap, τ_bg = τ_bg, K = K)

    if input_dir == ""
        final_cloud = evolve(conditions, duration;
            Nc = Nc, max_dt = max_dt, measure = measure)
    else
        sensor = loadsensor("$input_dir/sensor-data.csv")
    end

    # Save sensor data (before plotting, in case plotting fails)
    ft = filetime()
    dir = "./results/$ft-harmonic-verification"
    mkpath(dir)

    savecsv(sensor, "$dir/sensor-data.csv")

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

    window_time = min(4e-2, duration/2)
    time = sensor.time
    window_size = OptEvapCool.window_time_size(time, window_time)
    rolling_time = rollmean(time, window_size)

    # Colour palette
    col1 = RGB(0.0588, 0, 0.5882)
    col2 = RGB(0.9000, 0, 0.3490)
    col3 = RGB(1, 0.6510, 0)
    col4 = RGB(0.0824, 0.8392, 0)

    # Temperature
    rolling_temp = 2 / (3 * kB) * rollmean(sensor.ke, window_size)

    temp_order = -6
    temp_yformatter(y) = @sprintf("%.1f",y/(10.0^temp_order))
    
    temperature_plt = plot(rolling_time, rolling_temp,
        xlabel = "Time (s)",
        ylabel = L"\textrm{Temperature\ \ }(\mu K)",
        label = false,
        linecolor = col1,
        ls = :solid,
        #ylims = (0, Inf),
        minorticks = 5,
        yformatter = temp_yformatter,
        dpi = 300)
    #plot!(peth_t, peth_T, label = false, linecolor = col2, ls = :dash)
    plot!(pur_t, pur_T, label = false, linecolor = col3, ls = :dash)
    
    # Number
    Np_series = sensor.Nt .* sensor.F

    num_order = 6
    num_yformatter(y) = @sprintf("%.1f",y/(10.0^num_order))

    number_plt = plot(time, Np_series,
        xlabel = "Time (s)",
        ylabel = L"\textrm{Number\ \ }({}\times10^{%$num_order})",
        label = false,
        linecolor = col1,
        ls = :solid,
        #ylims = (0, Inf),
        minorticks = 5,
        yformatter = num_yformatter,
        dpi = 300)
    #plot!(peth_t, peth_N, label = false, linecolor = col2, ls = :dash)
    plot!(pur_t, pur_N, label = false, linecolor = col3, ls = :dash)

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
        ylims = (250, 350),
        ls = :solid,
        dpi = 300)
    #plot!(peth_t, peth_Γ, label = false, linecolor = col2, ls = :dash)
    plot!(pur_t, pur_Γ, label = false, linecolor = col3, ls = :dash)

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
        ls = :solid,
        ylims = (0, Inf),
        minorticks = 5,
        dpi = 300)

    # Save plots
    savefig(temperature_plt, "$dir/temp.png")
    savefig(collrate_plt, "$dir/collrate.png")
    savefig(number_plt, "$dir/number.png")
    savefig(psd_plt, "$dir/psd.png")

    return nothing
end