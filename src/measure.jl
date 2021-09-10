# Performing and plotting measurements during the simulation
using Statistics: mean

# Assumes the constant 'kB' is defined.

# Compute translational kinetic energy
function avg_kinetic_energy(velocities, m)
    # The extra 3 is to account for the fact that the velocities
    # array has 3xN elements, not just N.
    return 0.5 * m * 3 * mean(velocities .^2)
end

function avg_potential_energy(positions, potential, m, t)
    return mean(potential(positions, m, t))
end

# Record time, kinetic and potential energy, and collision counts
function record(time_series, ke_series, pe_series,
                test_counts, coll_counts, m, potential)

    return function(positions, velocities, tested_count, coll_count, t)
        push!(time_series, t)
        push!(ke_series, avg_kinetic_energy(velocities, m))
        push!(pe_series, avg_potential_energy(positions, potential, m, t))
        push!(test_counts, tested_count)
        push!(coll_counts, coll_count)
    end

end