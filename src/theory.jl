# HARMONIC APPROXIMATION THEORY

# Collision rate per atom
function equilibrium_collrate(field::HarmonicField, species, Np, T, t)
    ω_x, ω_y, ω_z = field.ωx(t), field.ωy(t), field.ωz(t)
    n̄ = Np * ω_x * ω_y * ω_z * (species.m / (4π * kB * T))^1.5
    v̄ = sqrt(8 * kB * T / (π * species.m))
    Γₐ = sqrt(2) * n̄ * species.σ * v̄
    return Γₐ
end

# Speed distribution
function equilibrium_speeds(m, T, max_speed = 0.01)
    N₀ = 4π * (m / (2π * kB * T))^1.5 # Harmonic normalisation constant
    f(v) = N₀ * v^2 * exp(-0.5 * m * v^2 / (kB * T)) # Ideal distribution
    speed_domain = range(0, stop=1.2 * max_speed, length=300)
    speed_density = f.(speed_domain)
    return speed_domain, speed_density
end

# Harmonic theory
function harmonic_theory(species, ωx, ωy, ωz, T₀, N₀, trap_depth, duration, τ_bg)
    dt = 0.000001

    ωx, ωy, ωz = time_parametrize(ωx, ωy, ωz)

    iter_count = ceil(Int64, duration / dt)
    N_series = zeros(Float64, iter_count)
    T_series = zeros(Float64, iter_count)
    Γ_series = zeros(Float64, iter_count)
    t_series = zeros(Float64, iter_count)
    #γ_series = zeros(Float64, iter_count)
    #depth_series = zeros(Float64, iter_count)

    m = species.m
    a = species.aₛ

    t = 0
    N = N₀
    T = T₀
    i = 1
    while i <= iter_count
        Γcoeff = 8*sqrt(2)*a^2 * m / (π * kB) * ωx(t) * ωy(t) * ωz(t)
        Γ =  Γcoeff * N / T # Peak collision rate (per atom?)
        εₜ = trap_depth(t)
        η = εₜ / (kB * T) # Truncation parameter
        Ṅₑ = - N * Γ * η * exp(-η) # Evaporation rate
        Ṅₗ = - N / τ_bg # Background loss rate
        Ṅ = Ṅₑ + Ṅₗ # Total removal rate
        ε̄ = 3 * kB * T # Average energy
        γ = (εₜ / ε̄ - 1) * (Ṅₑ / Ṅ) # Evaporation efficiency
        Ṫ = T * γ * (Ṅ / N) # Rate of temperature change

        dN = Ṅ * dt
        dT = Ṫ * dt

        N_series[i] = N
        T_series[i] = T
        Γ_series[i] = Γ / (2 * sqrt(2)) # Convert to AVG collrate
        t_series[i] = t

        N += dN
        T += dT
        t += dt

        i += 1
    end

    return t_series, N_series, T_series, Γ_series
end

function purdue_theory(species, ωx, ωy, ωz, T₀, N₀, trap_depth, duration, τ_bg = Inf, K = 0)
    dt = 0.000001

    ωx, ωy, ωz = time_parametrize(ωx, ωy, ωz)
    trap_depth = time_parametrize(trap_depth)

    iter_count = ceil(Int64, duration / dt)
    N_series = zeros(Float64, iter_count)
    T_series = zeros(Float64, iter_count)
    Γ_series = zeros(Float64, iter_count)
    t_series = zeros(Float64, iter_count)

    m = species.m
    σ = species.σ

    E₀ = 3 * N₀ * kB * T₀

    Γ_bg = 1 / τ_bg

    t = 0
    N = N₀
    E = E₀
    T = T₀

    i = 1
    while i <= iter_count
        n₀ = N * ωx(t) * ωy(t) * ωz(t) * (m / (2π * kB * T))^1.5 # Peak density
        n̄ = n₀ / sqrt(8) # Mean density
        vᵣ = sqrt(16 * kB * T / (π * m)) # Mean relative velocity

        εₜ = trap_depth(t) # Trap depth
        η = εₜ / (kB * T) # Truncation parameter
        Γ_el = n̄ * σ * vᵣ; # Elastic collision rate per atom
        Γ_ev = (η-4)*exp(-η) * Γ_el # Evaporation rate per atom
        Γ_3B = K * n₀^2 / (3 * sqrt(3)) # Three-body collision rate per atom

        Ṅ = - (Γ_ev + Γ_bg + Γ_3B) * N
        Ė_ev = - N * Γ_ev * (η + (η-5)/(η-4)) * kB * T
        Ė_bg = - Γ_bg * E
        Ė_3B = - Γ_3B * (2/3) * E

        Ė = Ė_ev + Ė_bg + Ė_3B

        # Note: ignoring effects of adiabatic change in trap shape

        dN = Ṅ * dt
        dE = Ė * dt

        N_series[i] = N
        T_series[i] = T
        Γ_series[i] = Γ_el
        t_series[i] = t

        N += dN
        E += dE
        T = E / (3 * N * kB)
        t += dt

        i += 1
    end

    return t_series, N_series, T_series, Γ_series
end