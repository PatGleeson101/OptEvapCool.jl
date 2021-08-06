#= Simulation initialisation:
- Particle distribution
- Potential / Acceleration

Atoms are represented by two (3 x N) arrays: one containing
position components, and the other velocity components.
=#

"""
Initialise N atoms with position components in the range
[-size, size] and velocity components in the range [-v, v],
both uniformly distributed. Also returns the indices of still-present
atoms, which initially is all of them.
"""
function init_uniform(N, size, v)
    positions = 2.0 * size * (rand(Float64, 3, N) .- 0.5)
    velocities = 2.0 * v * (rand(Float64, 3, N) .- 0.5)
    return positions, velocities
end

# Create harmonic acceleration and potential functions
# at the provided trap frequencies. These functions operate on
# several atoms at once, and have a dummy time-dependence.
function harmonic(ωx, ωy, ωz)
    ω_squared = [ωx^2, ωy^2, ωz^2]
    # Acceleration
    accel(positions, t) = (- ω_squared .* positions)
    # Potential of each particle, _per mass_
    potential(positions, t) = (
        0.5 * sum(ω_squared .* (positions .^2), dims = 1)
    )
    return accel, potential
end
