# DSMC Component
using StatsBase: samplepair, percentile
using LinearAlgebra: norm
using Random: MersenneTwister
using Printf: @sprintf
using ProgressMeter: Progress, update!
using Dates: now

# Grid structure
struct Grid
    cell_size # :: Vector{Float64}
    lower_corner # :: Vector{Float64}
    shape # :: Vector{Int64}
    cell_count # :: Int64
    cell_occupancies # :: Vector{Int64}
    cell_indices # :: Vector{Int64}
    assigned_atoms # :: Vector{Int64}

    function Grid(csize, lpos, shape, occupancies, indices, assignment)
        ccount = prod(shape)
        return new(csize, lpos, shape, ccount, occupancies,
                   indices, assignment)
    end
end

# Velocty Verlet step
function verlet_step!(positions, velocities, accel, t, dt)
    # Half timestep
    half_dt = dt / 2.0
    # Approximate position half-way through time-step
    positions .+= half_dt .* velocities
    # Acceleration halfway through timestep.
    velocities .+= dt .* accel(positions, t + half_dt)
    # Updated positions at end of timestep
    positions .+= half_dt .* velocities
end

# Assign particles to cells
function assign_cells(positions, peak_free_path, Nc)
    N = size(positions, 2)
    # Minimum and maximum coordinates for the bounding box around all atoms
    min_pos = minimum(positions, dims=2)
    max_pos = maximum(positions, dims=2)
    # Cell dimensions
    V = prod(max_pos .- min_pos)
    Vc = V * Nc / N
    ds = min(cbrt(Vc), peak_free_path) #NOTE: removed 1e-12 lower bound.
    cell_size = [ds, ds, ds];
    # Extend grid slightly beyond the furthest atoms, to
    # avoid edge atoms being placed outside the grid.
    lower_pos = min_pos - (0.25 * cell_size)
    upper_pos = max_pos + (0.25 * cell_size)
    grid_shape = ceil.(Int64, (upper_pos - lower_pos) ./ cell_size)
    cell_count = prod(grid_shape)
    cell_occupancies = zeros(Int64, cell_count)
    cell_indices = zeros(Int64, cell_count)
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
    atoms_by_cell = zeros(Int64, N)
    for atom in 1:N
        cell = assignments[atom]
        atoms_by_cell[ cell_indices[cell] ] = atom
        cell_indices[cell] += 1
    end
    # Restore cell_indices
    cell_indices .-= cell_occupancies

    nonempty = (!iszero).(cell_occupancies)
    peak_density = percentile(cell_occupancies[nonempty], 90) / Vc

    return Grid(cell_size, lower_pos, grid_shape, cell_occupancies,
                cell_indices, atoms_by_cell), peak_density
end

# Placeholder collision step
function collision_step!(velocities, dt, grid, σ, F, rng=MersenneTwister())
    # Compute maximum speed in each cell
    cell_count = grid.cell_count
    Ncs = grid.cell_occupancies
    squared_speeds = sum(velocities .* velocities, dims=1)
    max_speeds = zeros(cell_count)
    for cell in 1:cell_count
        Nc = Ncs[cell]
        if Nc > 0
            cell_index = grid.cell_indices[cell]
            # Store maximum speed
            max_speeds[cell] = sqrt(
                maximum(squared_speeds[cell_index:cell_index + Nc - 1])
            )
        end
    end
    # Calculate number of pairs to check in each cell
    dV = prod(grid.cell_size) # Cell volume
    M_unrounded = F * Ncs .* (Ncs .- 1) .* (dt * σ / dV) .* max_speeds;
    M_cand = ceil.(Int64, M_unrounded) # Number of candidate pairs in each cell.
    # Adjust probabilities to account for rounding
    probability_factors = M_unrounded ./ M_cand

    # Pre-compute random candidate pairs and corresponding check numbers
    M_cand_total = sum(M_cand)
    candidate_rands = rand(rng, M_cand_total)

    # Loop over cells; check M pairs in each
    cell_collisions = zeros(Int64, cell_count)
    cell_candidates = zeros(Int64, cell_count)
    # block_offset = 0
    for cell in 1:cell_count
        Nc = Ncs[cell]
        (Nc < 2) && continue # Skip cells with 0 or 1 particles
        # Check collision pairs
        start_index = grid.cell_indices[cell]
        Mc = M_cand[cell]
        probability_coefficient = 1 / max_speeds[cell] * probability_factors[cell];
        for _ in 1:Mc
            i, j = samplepair(rng, Nc) .+ (start_index - 1)
            atom1 = grid.assigned_atoms[i]
            atom2 = grid.assigned_atoms[j]
            # Get current velocities
            u1 = velocities[:,atom1];
            u2 = velocities[:,atom2];
            rel_speed = norm(u2 - u1)
            P_coll = rel_speed * probability_coefficient
            if candidate_rands[(cell_candidates[cell] += 1)] < P_coll
                cell_collisions[cell] += 1
                # Compute new relative velocity
                ϕ = 2 * π * rand(rng)
                cosθ = 2 * rand(rng) - 1
                sinθ = sqrt(1 - cosθ^2)
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

    # Calculate new maximum timestep based on collision rate
    nonempty = (!iszero).(Ncs)
    # Provided N != 0, there will be at least one nonempty cell
    colls_per_atom = cell_collisions[nonempty] ./ Ncs[nonempty]
    peak_colls_per_atom = percentile(colls_per_atom, 50)
    lower_coll_time = dt / (2 * peak_colls_per_atom)
    dt_new = 0.5 * lower_coll_time # Half of median collision time
    # If peak collisions per atom is zero, then dt_new is Infinity, and the
    # timestep will thus be limited by the other constraints (motion/trapping)

    return dt_new, sum(cell_candidates), sum(cell_collisions)
end

# Atom loss effects: high-energy particles, three-body recombination &
# background collisions.
function atom_loss_step!(positions, velocities, m, ε, dt)
    # Perfect loss model
    N0 = size(positions, 2)
    N = N0
    for i in N0:-1:1
        v = velocities[:,i]
        ke = 0.5 * m * v .^2
        if ke > ε
            # Remove atom by replacing it with atom from the end
            velocities[:,i] = velocities[:,N]
            positions[:,i] = positions[:,N]
            N -= 1
        end
    end

    return N
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

# Estimate for an appropriate motion timestep
function motion_timestep(positions, velocities, accel, t, motion_limit)
    accels = sqrt.(sum(accel(positions, t) .^2, dims=1))
    speeds = sqrt.(sum(velocities .^2, dims=1))
    motion_timesteps = vec(motion_limit ./ (accels .* speeds))
    return percentile(motion_timesteps, 10)
    # Note that infinities (if certain particles have a=0 or v=0)
    # are handled correctly by the percentile function (but of course
    # won't be in the bottom 10th percentile)
end

# Dummy measure function if no measurement function provided
null_measure(_...) = nothing

# Evolve initial particle population for desired duration
function evolve(positions, velocities, accel, duration, σ, ω_max, m, F = 1, Nc = 1,
                measure = null_measure, rng=MersenneTwister())

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
    # Repopulator needs access to the underlying storage arrays
    repopulate! = repopulator(position_array, velocity_array)

    # Dynamic quantities
    t = 0 # Time
    Nt = Nt_max # Current number of test particles
    # Note: Nc is the target for average # atoms per cell.
    # Initial peak density (1e-5 is just a generous upper bound on cell width)
    _, peak_density = assign_cells(position_array, 1e-5, Nc)
    # Proportionality constant relating timestep with max. acceleration
    motion_limit = 0.1
    # Maximum timestep based on trap frequency
    trap_dt = 0.1 * 2π/ω_max
    # Initial timestep based on trap frequency and motion limit
    dt = min(0.0001, trap_dt,
             motion_timestep(positions, velocities, accel, t, motion_limit))
    dt = 0.0001
    # Track progress
    prog_detail = 10000
    progress = Progress(prog_detail, dt = 10, desc = "Simulation progress: ",
                color = :green, showspeed = false, enabled = true, barlen=50)
    
    # Iterate simulation
    iter_count = 0
    start_time = now()
    while t < duration
        # Active particles
        positions = @view positions[:, 1:Nt]
        velocities = @view velocities[:, 1:Nt]
        
        verlet_step!(positions, velocities, accel, t, dt) # Collisionless motion
        # Sort atoms by cell and apply collisions
        peak_free_path = 1 / (4 * peak_density * σ);
        grid, peak_density = assign_cells(positions, peak_free_path, Nc)
        t += dt # Increment time before timestep is updated by collision stage
        coll_dt, cand_count, coll_count = collision_step!(velocities, dt, grid, σ, F, rng)
        motion_dt = motion_timestep(positions, velocities, accel, t, motion_limit)
        #= Ensure dt smaller than:
        - min trapping time
        - motion time based on accelerations and speeds
        - collision time
        =#
        dt = min(coll_dt, trap_dt, motion_dt)
        dt = max(dt, 0.00005)
        dt = 0.0001

        # N = atom_loss_step!(positions, velocities, m, dt)
        if Nt < Nt_max / 2
            Nt = repopulate!(positions, velocities)
        end
        # External measurements on system after one full iteration
        measure(positions, velocities, cand_count, coll_count, t)

        #Update progress
        if mod(iter_count, 100) == 0
            update!(progress, ceil(Int, t / duration * prog_detail),
                    showvalues = [
                        ("Virtual time", @sprintf("%.3g / %.3g s", t, duration)),
                        ("Real time", now() - start_time),
                        ("Iterations", iter_count),
                        ("Coll. dt", @sprintf("%.3g", coll_dt)),
                        ("Motion dt", @sprintf("%.3g", motion_dt)),
                        ("Trap dt", @sprintf("%.3g", trap_dt)),
                        ("Cell ds", @sprintf("%.3g", grid.cell_size[1])),
                        ("Grid shape", @sprintf("%i x %i x %i", grid.shape...)),
                        ("Cell count", prod(grid.shape)),
                        ("Non-empty", sum((!iszero).(grid.cell_occupancies))),
                        ("Peak density", peak_density)
                        ])
        end

        iter_count += 1
    end

    update!(progress, ceil(Int, t / duration * prog_detail),
                    showvalues = [
                        ("Virtual time", @sprintf("%.3g / %.3g s", t, duration)),
                        ("Real time", now() - start_time),
                        ("Iterations", iter_count),
                        ("Coll. dt", @sprintf("%.3g", coll_dt)),
                        ("Motion dt", @sprintf("%.3g", motion_dt)),
                        ("Trap dt", @sprintf("%.3g", trap_dt)),
                        ("Cell ds", @sprintf("%.3g", grid.cell_size[1])),
                        ("Grid shape", @sprintf("%i x %i x %i", grid.shape...)),
                        ("Cell count", prod(grid.shape)),
                        ("Non-empty", sum((!iszero).(grid.cell_occupancies))),
                        ("Peak density", peak_density)
                        ])

    # Return final positions and velocities
    return positions, velocities
end