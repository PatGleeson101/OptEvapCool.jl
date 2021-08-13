#= Performing and plotting measurements during the simulation
- Temperature, etc.
=#

using Statistics: mean

const kB = 1.38064852e-23 #Boltzmann constant

# Compute massic translational kinetic energy
function massic_kinetic_energy(velocities)
    # The extra 3 is to account for the fact that the velocities
    # array has 3xN elements, not just N.
    return 0.5 * 3 * mean(velocities .* velocities)
end

function massic_potential_energy(positions, potential, t)
    return mean(potential(positions, t))
end

function avg_temperature(velocities, m)
    avg_ke = m * massic_kinetic_energy(velocities)
    return 2 * avg_ke / (3 * kB)
end

# Generate a function to record temperature and time in the provided arrays
function record_temperature(temp_array, time_array, m)
    return function(positions, velocities, t)
        push!(temp_array, avg_temperature(velocities, m))
        push!(time_array, t)
    end
end

function record_massic_energy(energy_array, time_array, potential)
    return function(positions, velocities, t)
        ke = massic_kinetic_energy(velocities)
        pe = massic_potential_energy(positions, potential, t)
        push!(energy_array, ke + pe)
        push!(time_array, t)
    end
end

# Key function: records several quantities.
function record_all(time_array, ke_array, pe_array, dist_array,
                    temp_array, potential, start_positions, m,
                    test_particle_array)
    initial_positions = copy(start_positions)
    return function(positions, velocities, t)
        push!(time_array, t)
        push!(ke_array, massic_kinetic_energy(velocities))
        push!(pe_array, massic_potential_energy(positions, potential, t))
        displacements = positions - initial_positions
        avg_distance = mean(sqrt.(sum(displacements .^2, dims = 1)))
        push!(dist_array, avg_distance)
        push!(temp_array, avg_temperature(velocities, m))
        push!(test_particle_array, positions[:,1])
    end
end