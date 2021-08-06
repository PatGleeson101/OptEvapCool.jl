#= Running the DSMC:
- Motion step (Verlet)
- Collision step
=#

# Apply motion
function free_step!(positions, velocities, dt)
    positions .+= velocities * dt
end

# Velocty Verlet step, according to provided acceleration
function verlet(accel)
    return function(positions, velocities, t, dt)
        # Half timestep
        half_dt = dt / 2.0
        # Approximate position half-way through time-step
        positions .+= half_dt .* velocities
        # Acceleration halfway through timestep.
        velocities .+= dt .* accel(positions, t + half_dt)
        # Updated positions at end of timestep
        positions .+= half_dt .* velocities
    end
end

# Placeholder collision step
function collisions!(positions, velocities, atoms_by_cell)
    #TODO
end

# Atom loss effects: high-energy particles, three-body recombination &
# background collisions.
function atomloss!(positions, velocities, atoms_by_cell)
    # TODO
    #Placeholder to return correct N with no atom loss.
    return length(atoms_by_cell)
end

# Assign particles to cells
function assigner(N_max, xcount, ycount, zcount)
    grid_shape = [xcount, ycount, zcount]
    cell_count = xcount * ycount * zcount
    # Pre-initialise re-usable storage. ASSUMES fixed grid shape.
    cell_occupancies = zeros(UInt64, grid_shape...)
    cell_start_indices = zeros(UInt64, cell_count)
    atom_assignments = zeros(UInt64, 3, N_max)
    flat_assignments = zeros(UInt64, N_max)

    return function(positions)
        N = size(positions, 2)
        active_assignments = @view atom_assignments[:, 1:N]
        # Minimum and maximum coordinates for bounding box around all atoms
        min_pos = minimum(positions, dims = 2)
        max_pos = maximum(positions, dims = 2)
        # NOTE: Following procedure breaks if the distribution is
        # degenerate along a dimension (min - max = 0).

        # Approximate cell dimensions:
        cell_size = (max_pos - min_pos) ./ grid_shape
        # Extend grid slightly beyond furthest atoms.
        # This avoids edge atoms being placed outside the grid.
        lower_pos = min_pos - (0.25 * cell_size)
        upper_pos = max_pos + (0.25 * cell_size)
        # Compute adjusted cell dimensions
        cell_size = (upper_pos - lower_pos) ./ grid_shape
        # Reset the number of atoms in each cell to 0
        cell_occupancies .= zeros(UInt64)
        # Get cell indices for each particle
        active_assignments = (
            ceil.(UInt64, (positions .- lower_pos) ./ cell_size )
        )
        # Flatten from Cartesian indices to linear indices. #TODO: verify.
        flat_assignments .= [ (ix + xcount * (iy - 1 + ycount * (iz - 1)))
            for (ix, iy, iz) in eachcol(active_assignments) ]
        # Count number of atoms in each cell
        for atom_index in 1:N
            cell_occupancies[ flat_assignments[atom_index] ] += 1
        end
        # Compute cumulative starting index for each cell
        cumulative_index = 1
        for i in 1:cell_count
            cell_start_indices[i] = cumulative_index
            cumulative_index += cell_occupancies[i]
        end
        # Place atoms in cells
        atoms_by_cell = zeros(UInt64, N)
        for atom_index in 1:N
            cell_index = flat_assignments[atom_index]
            atoms_by_cell[ cell_start_indices[cell_index] ] = atom_index
            cell_start_indices[cell_index] += 1
        end
        # Restore cell start indices
        cell_start_indices .-= cell_occupancies[:]

        return atoms_by_cell
    end
end

# Simulation
function evolve!(positions, velocities, accel, duration, dt, measure)
    # TODO: calculate appropriate N_max and grid size/shape
    N_max = size(positions, 2) #Maximum number of particles
    grid_shape = [10, 10, 10]
    # Initialise update functions
    verlet_step! = verlet(accel)
    assign_cells = assigner(N_max, grid_shape...)

    N = N_max # Remaining number of particles
    t = 0
    while t < duration
        active_positions = @view positions[:, 1:N]
        active_velocities = @view velocities[:, 1:N]
        # Collisionless motion
        verlet_step!(active_positions, active_velocities, t, dt)
        # Array containing atom indices, sorted by cell.
        # TODO: also need to return cell_occupancies and/or cell_start_indices
        atoms_by_cell = assign_cells(active_positions)
        # TODO: pass cell-assignment to measure function??
        measure(active_positions, active_velocities, t) # External measurements
        # Collisions
        collisions!(active_positions, active_velocities, atoms_by_cell)
        # Atom loss due to high energy and other effects. This modifies each
        # of active_positions and active_velocities. Immediately afterwards,
        # atoms_by_cell is invalid. The remaining number of atoms is returned.
        N = atomloss!(active_positions, active_velocities, atoms_by_cell)
        #=
        if N < N_max/2
            # Repopulate if too few atoms remain
        end
        =#
        t += dt # Increment time
    end
end