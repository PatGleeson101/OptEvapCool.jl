# DSMC
using StatsBase: samplepair, percentile
using LinearAlgebra: norm
using Random: MersenneTwister
using Printf: @sprintf
using ProgressMeter: Progress, update!
using Dates: now

# Storage of essential atom cloud data
mutable struct AtomCloud
    position_buffer :: Matrix{Float64}
    velocity_buffer :: Matrix{Float64}
    assignment_buffer :: Vector{Int64}
    cell_buffer :: Vector{Int64}
    occupancy_buffer :: Vector{Int64}

    positions :: SubArray
    velocities :: SubArray
    cell_offsets :: SubArray
    cell_occupancies :: SubArray

    cellsize :: Vector{Float64}

    function AtomCloud(positions, velocities)
        # Check that array sizes are 3 x Nt; (Nt = initial # test particles)
        num_components, Nt = size(positions)
        size(velocities) != (num_components, Nt) && throw(DimensionMismatch(
            "Velocity and position arrays must have the same size.")
        )
        (num_components != 3) && throw(ArgumentError(
            "Position and velocity must have three components.")
        )
        # Create atom buffers
        posbuf = copy(positions)
        velbuf = copy(velocities)
        pos = view(posbuf, :, 1:Nt)
        vel = view(velbuf, :, 1:Nt)
        assbuf = zeros(Int64, Nt)
        # Following buffers and data will be overwritten on first iteration
        celbuf = zeros(Int64, 1)
        occbuf = zeros(Int64, 1)
        offsets = view(celbuf, 1:1)
        occupancies = view(occbuf, 1:1)
        cellsize = [0.0, 0.0, 0.0]

        # Return cloud AND initial number of atoms
        return new(posbuf, velbuf, assbuf, celbuf, occbuf,
                   pos, vel, offsets, occupancies, cellsize), Nt
    end
end

# Velocty Verlet step
function verlet_step!(cloud, accel, t, dt)
    positions, velocities = cloud.positions, cloud.velocities
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
function assign_cells!(cloud, peak_free_path, Nc)
    positions, velocities = cloud.positions, cloud.velocities
    N = size(positions, 2)

    # Corners of the bounding box around all atoms
    minpos = minimum(positions, dims=2)
    maxpos = maximum(positions, dims=2)

    # Cell dimensions
    V = prod(maxpos - minpos)
    Vc = V * Nc / N
    ds = min(cbrt(Vc), peak_free_path) #NOTE: removed 1e-12 lower bound.
    cellsize = [ds, ds, ds]
    cloud.cellsize = cellsize

    # Extend grid slightly beyond the furthest atoms, to
    # avoid edge atoms being placed outside the grid.
    lowerpos = minpos - (0.25 * cellsize)
    upperpos = maxpos + (0.25 * cellsize)
    gridshape = ceil.(Int64, (upperpos - lowerpos) ./ cellsize)
    cellcount = prod(gridshape)

    # Ensure storage buffers are large enough.
    if cellcount > length(cloud.cell_buffer)
        cloud.cell_buffer = zeros(Int64, cellcount)
        cloud.occupancy_buffer = zeros(Int64, cellcount)
    end
    cell_offsets = view(cloud.cell_buffer, 1:cellcount)
    cell_occupancies = view(cloud.occupancy_buffer, 1:cellcount)
    fill!(cell_offsets, 0)
    fill!(cell_occupancies, 0)

    # Assign atoms to cells and count the number of atoms in each cell.
    assignments = view(cloud.assignment_buffer, 1:N)
    xcount, ycount, _ = gridshape
    for atom in 1:N
        x, y, z = ceil.(Int64, (positions[:,atom] - lowerpos) ./ cellsize)
        cell = x + xcount * ( (y - 1) + ycount * (z - 1) )
        assignments[atom] = cell
        cell_occupancies[cell] += 1
    end

    # Compute storage offset for each cell.
    offset = 1
    for cell in 1:cellcount
        Nc = cell_occupancies[cell]
        if Nc > 0
            cell_offsets[cell] = offset
            offset += Nc
        end
    end

    # Rearrange atoms to be in cell-order.
    # This temporarily modifies the cell offsets.
    sorted_positions = zeros(Float64, 3, N)
    sorted_velocities = zeros(Float64, 3, N)
    for atom in 1:N
        cell = assignments[atom]
        sorted_index = cell_offsets[cell]
        sorted_positions[:, sorted_index] = positions[:,atom]
        sorted_velocities[:, sorted_index] = velocities[:,atom]
        cell_offsets[cell] += 1
    end

    # Restore cell offsets and move empty cells to the end.
    nonempty_count = 0
    for cell in 1:cellcount
        Nc = cell_occupancies[cell]
        if Nc > 0
            cell_offsets[cell] -= Nc
            nonempty_count += 1
            # Use current cell to overwrite earliest empty cell
            cell_offsets[nonempty_count] = cell_offsets[cell]
            cell_occupancies[nonempty_count] = cell_occupancies[cell]
        end
    end

    # Record offsets and occupancies for non-empty cells
    cloud.cell_offsets = view(cloud.cell_buffer, 1:nonempty_count)
    cloud.cell_occupancies = view(cloud.occupancy_buffer, 1:nonempty_count)

    # Compute peak density
    peak_density = percentile(cloud.cell_occupancies, 95) / Vc

    return peak_density, gridshape
end

# Simulate collisions
function collision_step!(cloud, dt, σ, F, rng=MersenneTwister())
    Ncs = cloud.cell_occupancies
    offsets = cloud.cell_offsets
    cellcount = length(offsets)
    velocities = cloud.velocities

    # Compute maximum speed in each cell
    sq_speeds = sum(velocities .* velocities, dims=1) # Squared speeds
    max_speeds = sqrt.([
        maximum( sq_speeds[ offsets[cell] : offsets[cell] + Ncs[cell] - 1])
        for cell in 1:cellcount
    ])

    # Calculate number of pairs to check in each cell
    Vc = prod(cloud.cellsize) # Cell volume
    Mraw = F * (dt * σ / Vc) * Ncs .* (Ncs .- 1) .* max_speeds # Unrounded pairs
    Mcand = ceil.(Int64, Mraw) # Rounded number of pairs in each cell
    prob_adjustment = Mraw ./ Mcand # Account for rounding in probabilities

    # Pre-compute random numbers for checking collision success
    cand_success = rand(rng, sum(Mcand))

    # Check appropriate number of pairs in each cell
    Mcoll = zeros(Int64, cellcount)
    cand_counter = 0
    for cell in 1:cellcount
        Nc = Ncs[cell]
        (Nc < 2) && continue # Skip cells with 1 particle
        # Get cell info
        offset = offsets[cell]
        Mc = Mcand[cell]
        prob_coeff = prob_adjustment[cell] / max_speeds[cell]
        # Check Mc candidate pairs
        for _ in 1:Mc
            atom1, atom2 = samplepair(rng, Nc) .+ (offset - 1)
            # Get current velocities
            u1 = velocities[:,atom1]
            u2 = velocities[:,atom2]
            urel = norm(u2 - u1)
            pcoll = urel * prob_coeff # Collision probability
            if cand_success[cand_counter += 1] < pcoll
                # Compute new relative velocity
                ϕ = 2 * π * rand(rng)
                cosθ = 2 * rand(rng) - 1
                sinθ = sqrt(1 - cosθ^2)
                vrel = urel .* [sinθ * cos(ϕ), sinθ * sin(ϕ), cosθ]
                # Update stored velocities
                vcm = 0.5 * (u1 + u2)
                v1 = vcm + 0.5 * vrel;
                v2 = vcm - 0.5 * vrel;
                velocities[:,atom1] = v1;
                velocities[:,atom2] = v2;
                # Update cell's collision count and maximum speed
                max_speeds[cell] = max(max_speeds[cell], norm(v1), norm(v2))
                prob_coeff = prob_adjustment[cell] / max_speeds[cell]
                Mcoll[cell] += 1
            end
        end
    end

    # Calculate new maximum timestep based on collision rate
    colls_per_atom = Mcoll ./ Ncs
    peak_colls_per_atom = percentile(colls_per_atom, 95)
    lower_coll_time = dt / (2 * peak_colls_per_atom)
    # If peak collisions per atom is zero, then dt is Infinity, and the
    # timestep will be limited by the other constraints (motion/trapping)
    dt = 0.5 * lower_coll_time

    return dt, sum(Mcand), sum(Mcoll)
end

# Atom loss effects: high-energy particles, three-body recombination &
# background collisions.
function atom_loss_step!(cloud, m, ε, dt)
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

# Repopulate cloud by creating a duplicate of each
# particle, with the opposite velocity.
function duplicate!(cloud, F)
    positions = copy(cloud.positions)
    velocities = copy(cloud.velocities)
    N₀ = size(positions, 2)
    N₁ = 2 * N₀
    if N₁ > size(cloud.position_buffer, 2)
        # Resize buffer (only necessary if you increase # atoms)
        cloud.position_buffer = zeros(Float64, 3, N₀)
        cloud.velocity_buffer = zeros(Float64, 3, N₀)
        cloud.position_buffer[1:N₀] .= positions
        cloud.velocity_buffer[1:N₀] .= velocities
    end
    cloud.position_buffer[N₀+1 : N₁] .= positions
    cloud.velocity_buffer[N₀+1 : N₁] .= velocities
    return N₁, F/2
end

# Estimate for an appropriate motion timestep
function motion_timestep(cloud, accel, t, motion_limit)
    accels = sqrt.( sum( accel(cloud.positions, t).^2, dims=1) )
    speeds = sqrt.( sum(cloud.velocities .^2, dims=1) )
    motion_timesteps = vec(motion_limit ./ (accels .* speeds))
    return percentile(motion_timesteps, 5)
    # Note that infinities (if certain particles have a=0 or v=0)
    # are handled correctly by the percentile function (but of course
    # won't be in the bottom percentiles)
end

# Dummy measure function if no measurement function provided
null_measure(_...) = nothing

# Evolve initial particle population for desired duration
function evolve(positions, velocities, accel, duration, σ, ω_max, m, F = 1, Nc = 1,
                measure = null_measure, rng=MersenneTwister())
    # Note: Nc is the target for average # atoms per cell.
    # Initialise cloud
    cloud, Nt_max = AtomCloud(positions, velocities)

    # Dynamic quantities
    t = 0 # Time
    Nt = Nt_max # Current number of test particles

    # Initial peak density (1e-5 limits max cell width)
    peak_density, _ = assign_cells!(cloud, 1e-5, Nc)

    # Proportionality constant relating timestep with max. acceleration
    motion_limit = 0.01

    # Maximum timestep based on trap frequency
    trap_dt = 0.1 * 2π / ω_max

    # Initial timestep based on trap frequency and motion limit
    #dt = min(0.0001, trap_dt, motion_timestep(cloud, accel, t, motion_limit))
    dt = 0.0001

    # Track progress
    prog_detail = 10000
    progress = Progress(prog_detail, dt = 10, desc = "Simulation progress: ",
                color = :green, showspeed = false, enabled = true, barlen=50)
    
    # Iterate simulation
    iter_count = 0
    start_time = now()
    while t < duration
        # Collisionless motion
        verlet_step!(cloud, accel, t, dt)

        # Sort atoms by cell
        peak_free_path = 1 / (4 * peak_density * σ);
        peak_density, gridshape = assign_cells!(cloud, peak_free_path, Nc)

        t += dt # Increment time now, before dt is updated

        # Perform collisions and update timestep
        coll_dt, cand_count, coll_count = collision_step!(cloud, dt, σ, F, rng)
        motion_dt = motion_timestep(cloud, accel, t, motion_limit)
        #dt = min(coll_dt, trap_dt, motion_dt)
        dt = 0.0001

        # Nt = atom_loss_step!(cloud, m, dt)
        if Nt < Nt_max / 2
            Nt, F = duplicate!(cloud, F)
        end

        # External measurements on system after one full iteration
        measure(positions, velocities, cand_count, coll_count, t)

        #Update progress
        if mod(iter_count, 100) == 0
            update!(progress, ceil(Int64, prog_detail * t / duration),
                showvalues = [
                ("Virtual time", @sprintf("%.3g / %.3g s", t, duration)),
                ("Real time", now() - start_time),
                ("Iterations", iter_count),
                ("Coll. dt", @sprintf("%.3g", coll_dt)),
                ("Motion dt", @sprintf("%.3g", motion_dt)),
                ("Trap dt", @sprintf("%.3g", trap_dt)),
                ("Cell size", @sprintf("%.3g x %.3g x %.3g", cloud.cellsize...)),
                ("Grid shape", @sprintf("%i x %i x %i", gridshape...)),
                ("Non-empty cell count", length(cloud.cell_offsets)),
                ("Peak density", peak_density)
                ])
        end

        iter_count += 1
    end

    # Final update
    update!(progress, ceil(Int64, prog_detail * t / duration),
        showvalues = [
        ("Virtual time", @sprintf("%.3g / %.3g s", t, duration)),
        ("Real time", now() - start_time),
        ("Iterations", iter_count),
        ])

    # Return final positions and velocities
    return cloud.positions, cloud.velocities
end