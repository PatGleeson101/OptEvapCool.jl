# HARMONIC APPROXIMATION THEORY

# Collision rate per atom
function equilibrium_collrate(field::HarmonicField, species, Np, T, t)
    ω_x, ω_y, ω_z = field.ωx(t), field.ωy(t), field.ωz(t)
    n̄ = Np * ω_x * ω_y * ω_z * (species.m / (4π * kB * T))^1.5
    v̄ = sqrt(8 * kB * T / (π * species.m))
    #Γ = 1 / sqrt(2) * Np * n̄ * species.σ * v̄ #Total collision rate
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
    dt = 0.0001

    iter_count = ceil(Int64, duration / dt)
    N_series = zeros(Float64, iter_count)
    T_series = zeros(Float64, iter_count)
    Γ_series = zeros(Float64, iter_count)
    t_series = zeros(Float64, iter_count)
    #γ_series = zeros(Float64, iter_count)
    #depth_series = zeros(Float64, iter_count)

    m = species.m
    a = species.aₛ

    Γcoeff = 8*sqrt(2)*a^2 * m / (π * kB) * ωx * ωy * ωz

    t = 0
    N = N₀
    T = T₀
    i = 1
    while i <= iter_count
        Γ =  Γcoeff * N / T # Peak collision rate
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
        Γ_series[i] = Γ
        t_series[i] = t

        N += dN
        T += dT
        t += dt

        i += 1
    end

    return t_series, N_series, T_series, 3*Γ_series #TODO: REMOVE FUDGE FACTOR
end