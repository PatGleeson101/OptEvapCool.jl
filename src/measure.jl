#= Performing and plotting measurements during the simulation
- Temperature, etc.
=#

using Statistics: mean

const kB = 1.38064852e-23 #Boltzmann constant

# Compute total translational kinetic energy
function total_kinetic_energy(velocities, m)
    return 0.5 * m * sum(velocities .* velocities)
end

function total_potential_energy(positions, potential, t, m)
    return m * sum(potential(positions, t))
end

function avg_temperature(velocities, m)
    K = total_kinetic_energy(velocities, m)
    return 2 * K / (3 * size(velocities, 2) * kB)
end

# Generate a function to record temperature and time in the provided arrays
function record_temperature(temp_array, time_array, m)
    return function(positions, velocities, t)
        push!(temp_array, avg_temperature(velocities, m))
        push!(time_array, t)
    end
end

function record_total_energy(energy_array, time_array, potential, m)
    return function(positions, velocities, t)
        ke = total_kinetic_energy(velocities, m)
        pe = total_potential_energy(positions, potential, t, m)
        push!(energy_array, ke + pe)
        push!(time_array, t)
    end
end

# Key function: records several quantities.
function record_all(time_array, ke_array, pe_array, dist_array,
                    temp_array, potential, start_positions, m)
    initial_positions = copy(start_positions)
    return function(positions, velocities, t)
        push!(time_array, t)
        push!(ke_array, total_kinetic_energy(velocities, m))
        push!(pe_array, total_potential_energy(positions, potential, t, m))
        displacements = positions - initial_positions
        avg_distance = mean(sqrt.(sum(displacements .^2, dims = 1)))
        push!(dist_array, avg_distance)
        push!(temp_array, avg_temperature(velocities, m))
    end
end