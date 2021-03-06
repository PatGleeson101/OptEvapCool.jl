using OptEvapCool
using StatsPlots
using Printf: @sprintf
using LaTeXStrings
using Plots.PlotMeasures

function otago_ghost_trap(duration = 0.24, input_dir = "")
    # Physical parameters
    Np = 1.6e6 # Initial atom count
    T₀ = 2e-6 # Initial temperature
    species = Rb87

    # Numerical parameters
    Nt = ceil(Int64, 1e5) # Test particles
    F = Np / Nt
    Nc = 3 # Target number of test particles per cell

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

    # Ghost beam focus
    foc_g = (t) -> [85e-6, 0, 0] + (t/0.24) * [-30e-6, 0, 0]

    beam_h = GaussianBeam([0,0,0], dir_h, P_h, w_h, λ)
    beam_v = GaussianBeam([0,0,0], dir_v, P_v, w_v, λ)
    beam_g = GaussianBeam(foc_g, dir_g, P_g, w_g, λ)

    # Trapping frequencies
    m = species.m
    κ = kappa(species)

    ωr²_v = 8κ*P_v / (m * π * w_v^4) # Radial trapping frequency
    ωr²_h = 8κ*P_h / (m * π * w_h^4)

    ωx = sqrt(ωr²_v + ωr²_h)
    ωy = sqrt(ωr²_h)
    ωz = sqrt(ωr²_v)

    # Cloud initialisation
    positions = harmonic_boltzmann_positions(Nt, m, T₀, ωx, ωy, ωz)
    velocities = boltzmann_velocities(Nt, m, T₀)

    # Translate to equilibrium position
    y₀ = - g / ωy^2
    centre = [0, y₀, 0]
    positions .+= centre

    # Three-body loss & background loss
    K = 4.3e-29 * 1e-12
    τ_bg = 180

    # Maximum timestep
    max_dt = 0.05 * 2π / max(ωx, ωy, ωz)

    # Gravity
    acc_grav! = acceleration(gravity)
    pot_grav! = potential(gravity)

    # GAUSSIAN BEAM SIMULATION
    acc_h! = acceleration(beam_h)
    acc_v! = acceleration(beam_v)
    acc_g! = acceleration(beam_g)

    pot_h! = potential(beam_h)
    pot_v! = potential(beam_v)
    pot_g! = potential(beam_g)

    function accel(p, s, t, o)
        fill!(o, 0.0)
        acc_h!(p, s, t, o)
        acc_v!(p, s, t, o)
        acc_g!(p, s, t, o)
        acc_grav!(p, s, t, o)
        return nothing
    end

    function poten(p, s, t, o)
        fill!(o, 0.0)
        pot_h!(p, s, t, o)
        pot_v!(p, s, t, o)
        pot_g!(p, s, t, o)
        pot_grav!(p, s, t, o)
        return nothing
    end

    sensor = GlobalSensor()
    measure = measurer(sensor, 0.00001, ωx, ωy, ωz, centre)

    T(t) = (length(sensor.ke) > 0) ? 2 * last(sensor.ke) / (3 * kB) : T₀
    evap = ellipsoid_evap(ωx, ωy, ωz, T, 1e-6, centre)

    conditions = SimulationConditions(species, F, positions, velocities,
        accel, poten, evap = evap, τ_bg = τ_bg, K = K)
    
    if input_dir == ""
        final_cloud = evolve(conditions, duration;
            Nc = Nc, max_dt = max_dt, measure = measure)
    else
        sensor = loadsensor("$(input_dir)/sensor-data.csv")
    end

    # Save sensor data (before plotting, in case plotting fails)
    ft = filetime()
    dir = "./results/$ft-otago-ghost-trap"
    mkpath(dir)
    savecsv(sensor, "$dir/sensor-data.csv")

    # PLOTTING
    default(fontfamily="Computer Modern",
        linewidth=2, framestyle=:box, label=nothing, grid=false)
    scalefontsizes()
    scalefontsizes(1.5)

    window_time = min(duration/2, 1e-2)
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
        ylims = (0, Inf),
        minorticks = 5,
        yformatter = temp_yformatter,
        dpi = 300)
    
    # Number
    instant_Np = sensor.Nt .* sensor.F

    num_order = 6
    num_yformatter(y) = @sprintf("%.2f",y/(10.0^num_order))

    number_plt = plot(time, instant_Np,
        xlabel = "Time (s)",
        ylabel = L"\textrm{Number\ \ }({}\times10^{%$num_order})",
        label = false,
        linecolor = col1,
        ls = :solid,
        ylims = (0, Inf),
        minorticks = 5,
        yformatter = num_yformatter,
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
        minorticks = 5,
        ls = :solid,
        dpi = 300)
    
    # Number vs PSD
    rolling_Np = rollmean(instant_Np, window_size)

    numpsd_plt = plot(rolling_psd, rolling_Np,
        xlabel = "Phase space density",
        ylabel = L"\textrm{Number\ \ }({}\times10^{%$num_order})",
        yformatter = num_yformatter,
        label = false,
        linecolor = col1,
        ls = :solid,
        ylims = (0, 1.8e6),
        #xlims = (0, 2.8),
        right_margin = 5mm,
        minorticks = 5,
        dpi = 300)
    scatter!([2.6], [1.4e5], color = "black", label = false,
        markersize = 8, markershape = :utriangle)
    #vline!([2.6], line = (:black, 5))

    # Save plots
    savefig(temperature_plt, "$dir/temp.png")
    savefig(collrate_plt, "$dir/collrate.png")
    savefig(number_plt, "$dir/number.png")
    savefig(psd_plt, "$dir/psd.png")
    savefig(numpsd_plt, "$dir/numpsd.png")

    return nothing
end