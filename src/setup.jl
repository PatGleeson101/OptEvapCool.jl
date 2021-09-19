# Simulation initialisation

using Distributions: Exponential, Bernoulli
using Random: rand!

#Boltzmann constant
const kB = 1.38064852e-23

# Defines a neutral atom species
struct AtomSpecies
    m :: Float64    # Atomic mass
    aₛ :: Float64   # Scattering length
    σ :: Float64    # Scattering cross-section
    α :: Float64    # Polarizability
end

"""
Initialise N atoms with position components in the range
[-size, size] and velocity components in the range [-v, v],
both uniformly distributed. Also returns the indices of still-present
atoms, which initially is all of them.
"""
function uniform_cube_cloud(N, size, v)
    # v: average speed in a single dimension
    # size: half-width of position cube
    positions = 2.0 * size * (rand(Float64, 3, N) .- 0.5)
    velocities = 2.0 * v * (rand(Float64, 3, N) .- 0.5)
    return positions, velocities
end

function boltzmann_velocities(N, m, T)
    beta = 2*kB*T/m# scale parameter
    speeds = sqrt.( rand( Exponential(beta), 3, N) )
    negate = rand( Bernoulli(0.5), 3, N) .* 2 .- 1
    velocities = (speeds .*= negate)
    return velocities
end

function harmonic_boltzmann_positions(N, m, T, ωx, ωy, ωz)
    beta_x, beta_y, beta_z = 2*kB*T/m ./ [ωx^2, ωy^2, ωz^2] # scale parameter
    distances = zeros(Float64, 3, N) # SQUARED distances
    rand!( Exponential(beta_x), @view distances[1, :])
    rand!( Exponential(beta_y), @view distances[2, :])
    rand!( Exponential(beta_z), @view distances[3, :])
    distances .= sqrt.( distances )
    negate = rand( Bernoulli(0.5), 3, N) .* 2 .- 1
    positions = (distances .*= negate)
    return positions
end

# Create harmonic acceleration and potential functions
# at the provided trap frequencies. These functions operate on
# several atoms at once, and have a dummy time-dependence.
function harmonic_field(ωx, ωy, ωz)
    ω_squared = [ωx^2, ωy^2, ωz^2]
    # Acceleration
    function accel(positions, t, out=copy(positions))
        out .= .- ω_squared .* positions
        return out
    end
    # Potential of each particle, _per mass_
    potential(positions, m, t) = (
        0.5 * m * sum(ω_squared .* (positions .^2), dims = 1)
    )
    return accel, potential
end

# TODO: function gaussian_beam()

function exponential_ramp(ϵ₀, τ)
    return (t) -> ϵ₀ * exp(- t / τ)
end
