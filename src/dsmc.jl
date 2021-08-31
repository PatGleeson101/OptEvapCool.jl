# DSMC Component

using StatsBase:samplepair
using LinearAlgebra
using Random

# Grid structure
struct Grid
    cell_size # :: Vector{Float64}
    lower_corner # :: Vector{Float64}
    shape # :: Vector{UInt64}
    cell_count # :: UInt64
    cell_occupancies # :: Vector{UInt64}
    cell_indices # :: Vector{UInt64}
    assigned_atoms # :: Vector{UInt64}

    function Grid(csize, lpos, shape, occupancies, indices, assignment)
        ccount = prod(shape)
        return new(csize, lpos, shape, ccount, occupancies,
                   indices, assignment)
    end
end

# Velocty Verlet step function, according to a provided acceleration
function verlet_stepper(accel)
    return function (positions, velocities, t, dt)
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

# Assign particles to cells
function assign_cells(positions, a_sc, peak_density)
    N = size(positions, 2)
    # Minimum and maximum coordinates for the bounding box around all atoms
    min_pos = minimum(positions, dims=2)
    max_pos = maximum(positions, dims=2)
    # Compute cell dimensions
    peak_free_path = 1 / (32π * peak_density * a_sc^2);
    ds = max(a_sc, min(maximum(max_pos) / 10, peak_free_path / 3)) # Roughly appropriate cell width
    cell_size = [ds, ds, ds];
    # Extend grid slightly beyond the furthest atoms, to
    # avoid edge atoms being placed outside the grid.
    lower_pos = min_pos - (0.25 * cell_size)
    upper_pos = max_pos + (0.25 * cell_size)
    grid_shape = ceil.(UInt64, (upper_pos - lower_pos) ./ cell_size)
    cell_count = prod(grid_shape)
    cell_occupancies = zeros(UInt64, cell_count)
    cell_indices = zeros(UInt64, cell_count)
    # Get 3D cell indices for each particle (zero-indexed for y and z)
    assignments_3D = (
        ceil.(UInt64, (positions .- lower_pos) ./ cell_size) .- [0, 1, 1]
    )
    # Flatten to linear indices
    xcount, ycount, _ = grid_shape
    index_coefficients = [1, xcount, xcount * ycount]
    assignments = sum(assignments_3D .* index_coefficients, dims=1)

    # Count number of atoms in each cell
    for atom in 1:N
        cell_occupancies[ assignments[atom] ] += 1
    end
    # Compute cumulative starting index for each cell
    cell_index = 1
    for cell in 1:cell_count
        cell_indices[cell] = cell_index
        cell_index += cell_occupancies[cell]
    end
    # Place atoms in cells. This temporarily modifies the cell_indices
    atoms_by_cell = zeros(UInt64, N)
    for atom in 1:N
        cell = assignments[atom]
        atoms_by_cell[ cell_indices[cell] ] = atom
        cell_indices[cell] += 1
    end
    # Restore cell_indices
    cell_indices .-= cell_occupancies

    peak_density = maximum(cell_occupancies) / prod(cell_size)

    return Grid(cell_size, lower_pos, grid_shape, cell_occupancies,
                cell_indices, atoms_by_cell), peak_density
end

# Placeholder collision step
function collision_step!(velocities, dt, grid, a_sc, F)
    # Compute maximum speed in each cell
    cell_count = grid.cell_count
    Ncs = grid.cell_occupancies
    squared_speeds = sum(velocities .* velocities, dims=1)
    max_speeds = zeros(cell_count)
    for cell in 1:cell_count
        Nc = Ncs[cell]
        if Nc > 0
            cell_index = grid.cell_indices[cell]
            # Store maximum SQUARED speed
            max_speeds[cell] = (
                maximum(squared_speeds[cell_index:cell_index + Nc - 1])
            )
        end
    end
    max_speeds .= sqrt.(max_speeds) # Take square root
    # Calculate number of pairs to check in each cell
    dV = prod(grid.cell_size) # Cell volume
    σ = 8 * pi * a_sc^2 # Total collision cross section
    M_unrounded = F * Ncs .* (Ncs .- 1) .* (dt * σ / dV) .* max_speeds;
    M_cand = ceil.(UInt64, M_unrounded) # Number of candidate pairs in each cell.
    # Adjust probabilities to account for rounding
    probability_factors = M_unrounded ./ M_cand

    # Pre-compute random candidate pairs and corresponding check numbers
    M_cand_total = sum(M_cand)
    candidate_rands = rand(M_cand_total)

    # Loop over cells; check M pairs in each
    collision_count = 0
    candidate_count = 0
    # block_offset = 0
    for cell in 1:cell_count
        Nc = Ncs[cell]
        (Nc < 2) && continue # Skip cells with 0 or 1 particles
        # Check collision pairs
        start_index = grid.cell_indices[cell]
        Mc = M_cand[cell]
        probability_coefficient = 1 / max_speeds[cell] * probability_factors[cell];
        for _ in 1:Mc
            i, j = samplepair(convert(Int64, Nc)) .+ (start_index - 1)
            atom1 = grid.assigned_atoms[i]
            atom2 = grid.assigned_atoms[j]
            # Get current velocities
            u1 = velocities[:,atom1];
            u2 = velocities[:,atom2];
            rel_speed = norm(u2 - u1)
            P_coll = rel_speed * probability_coefficient
            if candidate_rands[(candidate_count += 1)] < P_coll
                collision_count += 1
                # Compute new relative velocity
                ϕ = 2 * π * rand()
                cosθ = 2 * rand() - 1
                sinθ = sqrt.(1 - cosθ^2)
                v_rel = rel_speed .* [sinθ * cos(ϕ), sinθ * sin(ϕ), cosθ]
                # Update stored velocities
                v_cm = 0.5 * (u1 + u2)
                v1 = v_cm + 0.5 * v_rel;
                v2 = v_cm - 0.5 * v_rel;
                velocities[:,atom1] = v1;
                velocities[:,atom2] = v2;
                # Update maximum cell speed
                max_speeds[cell] = max(max_speeds[cell], norm(v1), norm(v2))
            end
        end
    end

    dt_new = dt # TODO: calculate new timestep

    return dt_new, candidate_count, collision_count
end

# Atom loss effects: high-energy particles, three-body recombination &
# background collisions.
function atom_loss_step!(positions, velocities, e_trap, dt)
    # Perfect loss model
    return size(positions, 2)
end

# Perform repopulation
function repopulator(position_array, velocity_array)
    return function (positions, velocities, N)
        # Create a duplicate of each particle, with the opposite velocity.
        N_new = 2 * N
        position_array[N + 1:N_new] = positions
        velocity_array[N + 1:N_new] = - velocities
        return N_new
    end
end

# Evolve initial particle population for desired duration
function evolve(positions, velocities, accel, duration, measure, a_sc, F)
    # Get maximum number of test particles and check array sizes
    num_components, Nt_max = size(positions)
    size(velocities) != (num_components, Nt_max) && throw(DimensionMismatch(
        "Velocity and position arrays must have the same size.")
    )
    (num_components != 3) && throw(ArgumentError(
        "Position and velocity must have three components.")
    )
    # Initialise storage
    position_array = copy(positions)
    velocity_array = copy(velocities)
    
    # Initialise update functions
    verlet_step! = verlet_stepper(accel)
    repopulate! = repopulator(position_array, velocity_array)

    # Dynamic quantities
    t = 0 # Time
    Nt = Nt_max # Current number of test particles
    dt = 0.0001 # Initial timestep
    peak_density = Nt / (maximum(position_array)^3) # Rough initial peak density estimate

    # Iterate simulation
    while t < duration
        # Active particles
        positions = @view positions[:, 1:Nt]
        velocities = @view velocities[:, 1:Nt]
        
        verlet_step!(positions, velocities, t, dt) # Collisionless motion
        # Sort atoms by cell and apply collisions
        grid, peak_density = assign_cells(positions, a_sc, peak_density)
        dt, cand_count, coll_count = collision_step!(velocities, dt, grid, a_sc, F)
        # N = atom_loss_step!(positions, velocities, dt)
        # Note: after atom loss, the atoms_by_cell assignment is invalid.
        if Nt < Nt_max / 2
            Nt = repopulate!(positions, velocities)
        end
        t += dt # Increment time
        # External measurements on system after one full iteration
        measure(positions, velocities, cand_count, coll_count, t)
    end

    # Return final positions and velocities
    return positions, velocities
end