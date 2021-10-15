# Initialising simulation conditions

using Distributions: Normal
using Random: rand!
using LinearAlgebra: normalize, dot, norm

# Constants (SI units)
const kB = 1.38064852e-23 #Boltzmann
const c = 3e8 # Speed of light (m/s)
const ε₀ = 8.854e-12 # Vacuum permittivity

# ATOM INITIALISATION

# Define a neutral atom species
struct AtomSpecies
    m :: Float64    # Atomic mass (kg)
    aₛ :: Float64   # Scattering length (m)
    σ :: Float64    # Scattering cross-section (m^2)
    α :: Float64    # Real part of polarizability
end

const Rb87 = AtomSpecies(1.454660e-25, 5e-9, 8*pi * (5e-9)^2,
                         1.1e-38)

# Dipole trap constant
kappa(species::AtomSpecies) = species.α / (2 * ε₀ * c)

# Initialise N (test) particles with position and velocity components
# distributed uniformly in the range [-size, size] and [-v, v], respectively.
function uniform_cloud(N, size, v)
    positions = 2.0 * size * (rand(Float64, 3, N) .- 0.5)
    velocities = 2.0 * v * (rand(Float64, 3, N) .- 0.5)
    return positions, velocities
end

# Initialise Boltzmann-distributed velocities at temperature
# T for N particles, each of mass m.
boltzmann_velocities(N, m, T) = reshape(rand( Normal(0, sqrt(kB * T / m)), 3*N), 3, N)

# Initialise Boltzmann-distributed positions in a harmonic trap at
# temperature T. Generates N particles, each of mass m.
function harmonic_boltzmann_positions(N, m, T, ωx, ωy, ωz)
    coeff = sqrt(kB * T / m)
    positions = zeros(Float64, 3, N)
    rand!( Normal(0, coeff / ωx), view(positions, 1, :) )
    rand!( Normal(0, coeff / ωy), view(positions, 2, :) )
    rand!( Normal(0, coeff / ωz), view(positions, 3, :) )
    return positions
end

# FIELD INITIALISATION

# Convert values to time-parametrised functions
function time_parametrize(value)
    f(_) = value
    f() = value
    return f
end

# Allow functions to be parametrised (for convenience)
function time_parametrize(func::Function)
    f(t) = func(t)
    function f()
        @warn "Time-dependent function given no argument. Assuming t = 0."
        return f(0)
    end
    return f
end

# Parametrise multiple functions/values at once (for convenience)
time_parametrize(args...) = [time_parametrize(arg) for arg in args]

# Harmonic potential (time-dependent)
struct HarmonicField
    # Time-parametrised trap frequencies
    ωx :: Function
    ωy :: Function
    ωz :: Function
    # Initialise
    HarmonicField(ωx, ωy, ωz) = new( time_parametrize(ωx, ωy, ωz)... )
end

# Gaussian beam (time-dependent)
struct GaussianBeam
    focus :: Function
    direction :: Function
    power :: Function
    waist :: Function
    wavelength :: Function

    function GaussianBeam(args...)
        focus, direction, power, waist, wavelength = time_parametrize(args...)
        unit_dir(t) = normalize(direction(t))
        return new(focus, unit_dir, power, waist, wavelength)
    end
end

# E.g. gravity
struct UniformField
    strength :: Function # Vector acceleration (m/s^2)
    origin :: Function
    UniformField(str, origin = [0,0,0]) = new( time_parametrize(str, origin)... )
end

const gravity = UniformField([0, -9.81, 0])

# Acceleration (constant field)
function acceleration(field::UniformField, positions, species, t, output)
    # 'species' parameter required to conform to correct call signature.
    Nt = size(positions, 2)
    str = field.strength(t)
    for atom in 1:Nt
        output[:, atom] .= str
    end
    return output
end

# Potential (constant field)
function potential(field::UniformField, positions, species, t, output)
    # 'species' parameter required to conform to correct call signature.
    Nt = size(positions, 2)
    strength = field.strength(t)
    origin = field.origin(t)
    for atom in 1:Nt
        pos = view(positions, :, atom)
        output[atom] = - species.m * dot(strength, pos - origin)
    end
    return output
end

# Overload accel/potential by generating new output array when none provided.
acceleration(f, p, s, t) = acceleration(f, p, s, t, zeros(Float64, size(p)...))
potential(f, p, s, t) = potential(f, p, s, t, zeros(Float64, size(p, 2)))
# Initialise a potential or acceleration function to
potential(f) = (args...) -> potential(f, args...) #pos, spec, t, out
acceleration(f) = (args...) -> acceleration(f, args...)

# Acceleration (harmonic field)
function acceleration(field::HarmonicField, positions, species, t, output)
    # 'species' parameter required to conform to correct call signature.
    Nt = size(positions, 2)
    ωx2, ωy2, ωz2 = field.ωx(t)^2, field.ωy(t)^2, field.ωz(t)^2
    for atom in 1:Nt
        output[1, atom] = - ωx2 * positions[1, atom]
        output[2, atom] = - ωy2 * positions[2, atom]
        output[3, atom] = - ωz2 * positions[3, atom]
    end
    return output
end

# Potential (harmonic field)
function potential(field::HarmonicField, positions, species, t, output)
    Nt = size(positions, 2)
    m = species.m
    ωx2, ωy2, ωz2 = field.ωx(t)^2, field.ωy(t)^2, field.ωz(t)^2
    for atom in 1:Nt
        x, y, z = view(positions, :, atom)
        output[atom] = 0.5 * m * (ωx2 * x^2 + ωy2 * y^2 + ωz2 * z^2)
    end
    return output
end

# Get parameters of a Gaussian beam
parameters(b::GaussianBeam, t) = ( b.focus(t), b.direction(t), b.power(t),
                                   b.waist(t), b.wavelength(t) )

# Acceleration (Gaussian beam)
function acceleration(field::GaussianBeam, positions, species, t, output)
    foc, dir, P, w₀, _ = parameters(field, t)
    κ = kappa(species)
    N = size(positions, 2)

    fx, fy, fz = foc
    ux, uy, uz = dir

    r²coeff = - 2 / w₀^2
    acoeff = 8 * P * κ / (π * species.m * w₀^4)
    Threads.@threads for atom in 1:N
        px, py, pz = view(positions, :, atom)
        dx = px - fx # Displacement from beam focus
        dy = py - fy
        dz = pz - fz
        z = dx*ux + dy*uy + dz*uz
        rx = dx - (z * ux)
        ry = dy - (z * uy)
        rz = dz - (z * uz)
        r² = rx^2 + ry^2 + rz^2
        a = (-acoeff * exp(r²coeff * r²))
        output[1, atom] = a*rx
        output[2, atom] = a*ry
        output[3, atom] = a*rz
    end
    return output
end

# Potential (Gaussian beam)
function potential(field::GaussianBeam, positions, species, t, output)
    foc, dir, P, w₀, _ = parameters(field, t)

    fx, fy, fz = foc
    ux, uy, uz = dir

    Nt = size(positions, 2)
    κ = kappa(species)
    r²coeff = - 2 / w₀^2
    Ucoeff = - κ * 2 * P / (π * w₀^2)
    Threads.@threads for atom in 1:Nt
        px, py, pz = view(positions, :, atom)
        dx = px - fx
        dy = py - fy
        dz = pz - fz
        z = dx*ux + dy*uy + dz*uz
        r² = (dx^2 + dy^2 + dz^2) - z^2
        output[atom] = Ucoeff * exp(r²coeff * r²) - Ucoeff
    end
    return output
end

# EVAPORATION PROBABILITIES

# An exponential ramp
exponential_ramp(start, stop, τ) = (t) -> stop + (start - stop) * exp(- t / τ)

# A linear ramp
linear_ramp(start, stop, τ) = (t) -> start + (stop - start) * (t / τ)

# Perfect energy-based removal
function energy_evap(εₜ, positions, velocities, conditions, potential, t)
    pe = potential(positions, conditions.species, t)
    ke = 0.5 * conditions.species.m * sum(velocities .* velocities, dims = 1)
    for atom in eachindex(pe)
        if pe[atom] + ke[atom] > εₜ
            pe[atom] = 1
        else
            pe[atom] = 0
        end
    end
    prob = pe # Evaporation probabilities
    return prob
end

# Overload for correct type signature and time-parametrisation
function energy_evap(depth, potential)
    εₜ = time_parametrize(depth)
    return (pos, vel, cond, t) -> energy_evap(εₜ(t), pos, vel, cond, potential, t)
end

# Removal when beyond ellipsoid
function ellipsoid_evap(ωx, ωy, ωz, T, p, centre, species, positions)
    N = size(positions, 2)

    m = species.m
    σx² = kB * T / (m * ωx^2)
    σy² = kB * T / (m * ωy^2)
    σz² = kB * T / (m * ωz^2)

    evap_prob = zeros(Float64, N)
    bound = -2 * log(p) # Boundary condition

    for atom in 1:N
        x, y, z = view(positions, :, atom) .- centre
        if (x^2 / σx² + y^2 / σy² + z^2 / σz²) > bound
            evap_prob[atom] = 1
        end
    end

    return evap_prob
end

# Overload for correct type signature and time-parametrisation
function ellipsoid_evap(ωx, ωy, ωz, T, p, centre = [0.0,0,0])
    ω_x, ω_y, ω_z = time_parametrize(ωx, ωy, ωz)
    centre = time_parametrize(centre)
    function evap(pos, vel, cond, t)
        return ellipsoid_evap(ω_x(t), ω_y(t), ω_z(t), T, p, centre(t), cond.species, pos)
    end
    return evap
end

# No evaporation (evap. probability 0)
no_evap(pos, args...) = zeros(Float64, size(pos, 2))

# SIMULATION CONDITIONS
struct SimulationConditions
    species :: AtomSpecies

    F :: Float64 # Initial F-scale
    positions :: Matrix{Float64} # Initial test-particle positions
    velocities :: Matrix{Float64} # Initial test-particle velocities

    acceleration :: Function # Time-parametrised, vectorised acceleration func.
    potential :: Function # Potential (also parametrised/vectorised)

    threebodyloss :: Float64 # 0 # 3-body loss constant (m^6/s, NOT cm^6/s)
    evaporate :: Function # no_evap # Particle loss-function
    τ_bg :: Float64 # Inf # Background loss time constant

    # Initialisation
    function SimulationConditions(spec, F, pos, vel, accel, potential;
            K = 0, evap = no_evap, τ_bg = Inf) # Optional arguments

        # Check that array sizes are 3 x Nt
        num_components, Nt = size(pos)
        size(vel) != (num_components, Nt) && throw(DimensionMismatch(
            "Velocity and position arrays must have the same size.")
        )
        (num_components != 3) && throw(ArgumentError(
            "Position and velocity must have three components.")
        )

        return new(spec, F, pos, vel, accel, potential, K, evap, τ_bg)        
    end
end