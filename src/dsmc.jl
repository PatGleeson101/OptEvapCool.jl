# DSMC
using StatsBase: samplepair, percentile
using LinearAlgebra: norm
using Random: MersenneTwister
using Printf: @sprintf
using ProgressMeter: Progress, update!, finish!
using Dates: now

# Storage of essential atom cloud data
mutable struct AtomCloud
    # BUFFERS
    positions :: Matrix{Float64}
    velocities :: Matrix{Float64}
    accels :: Matrix{Float64}
    assignments :: Vector{Int64}
    cell_offsets :: Vector{Int64}
    cell_occupancies :: Vector{Int64}
    atom_lookup :: Vector{Int64}

    Nt :: Int64
    F :: Float64
    cellcount :: Int64
    nonempty_count :: Int64
    cellsize :: Vector{Float64}

    function AtomCloud(positions, velocities, F)
        # Check that array sizes are 3 x Nt; (Nt = initial # test particles)
        num_components, Nt = size(positions)
        size(velocities) != (num_components, Nt) && throw(DimensionMismatch(
            "Velocity and position arrays must have the same size.")
        )
        (num_components != 3) && throw(ArgumentError(
            "Position and velocity must have three components.")
        )
        # Create atom buffers
        pos = copy(positions)
        vel = copy(velocities)
        acc = zeros(Float64, 3, Nt)
        assign = zeros(Int64, Nt)
        atoms = zeros(Int64, Nt)
        offsets = zeros(Int64, 1)     # To be overwritten
        occupancies = zeros(Int64, 1) # on first iteration

        cellsize = [0.0, 0.0, 0.0]
        nonempty_count = 1 # To be overwritten
        cellcount = 1      # on first iteration

        # Return cloud
        return new(pos, vel, acc, assign, offsets, occupancies, atoms,
                   Nt, F, cellcount, nonempty_count, cellsize)
    end
end

# Velocty Verlet step + calculate motion-based limit on next timestep
function verlet_step!(cloud, accel, t, dt, motion_limit)
    Nt = cloud.Nt
    positions = view(cloud.positions, :, 1:Nt)
    velocities = view(cloud.velocities, :, 1:Nt)
    accels = view(cloud.accels, :, 1:Nt)
    # Half timestep
    half_dt = dt / 2.0
    # Approximate position half-way through timestep
    for i in eachindex(positions) # Elementwise iteration
        positions[i] += half_dt * velocities[i]
    end
    # Acceleration halfway through timestep (modifies cloud.accels in-place)
    accel(positions, t + half_dt, accels)
    # Compute final velocities and positions, as well as new timestep
    min_timestep = Inf
    for atom in 1:Nt
        a = view(accels, :, atom)
        v = view(velocities, :, atom)   
        v .+= dt .* a # Update velocity
        positions[:, atom] .+= half_dt .* v # Update position
        speed = norm(v)
        acc = norm(a)
        timestep = motion_limit / (acc * speed)
        min_timestep = min(min_timestep, timestep)
    end

    return min_timestep
end

# Assign particles to cells
function assign_cells!(cloud, peak_free_path, Nc)
    N = cloud.Nt
    positions = view(cloud.positions, :, 1:N)

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
    if cellcount > length(cloud.cell_offsets)
        cloud.cell_offsets = zeros(Int64, cellcount * 2)
        cloud.cell_occupancies = zeros(Int64, cellcount * 2)
    end
    cell_offsets = view(cloud.cell_offsets, 1:cellcount)
    cell_occupancies = view(cloud.cell_occupancies, 1:cellcount)
    fill!(cell_offsets, 0)
    fill!(cell_occupancies, 0)

    # Assign atoms to cells and count the number of atoms in each cell.
    assignments = cloud.assignments
    xcount, ycount, _ = gridshape
    lx, ly, lz = lowerpos # Cache lower coordinates
    for atom in 1:N
        x = ceil(Int64, (positions[1, atom] - lx) / ds)
        y = ceil(Int64, (positions[2, atom] - ly) / ds)
        z = ceil(Int64, (positions[3, atom] - lz) / ds)
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

    # Create lookup index of atoms in each cell.
    # This temporarily modifies the cell offsets.
    atom_lookup = cloud.atom_lookup
    for atom in 1:N
        cell = assignments[atom]
        atom_lookup[ cell_offsets[cell] ] = atom
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

    cloud.nonempty_count = nonempty_count
    cloud.cellcount = cellcount

    # Compute peak density. Unlike collision rate, cell occupancies *need* to
    # be stored, so we might as well take a percentile.
    peak_density = percentile(cell_occupancies, 95) / Vc

    return peak_density, gridshape
end

# Simulate collisions
function collision_step!(cloud, dt, σ, rng=MersenneTwister())
    cellcount = cloud.nonempty_count
    Ncs = cloud.cell_occupancies
    offsets = cloud.cell_offsets
    velocities = cloud.velocities
    atom_lookup = cloud.atom_lookup
    F = cloud.F

    # Check appropriate number of pairs in each cell
    #Mcoll = zeros(Int64, cellcount)
    Vc = prod(cloud.cellsize) # Cell volume
    tot_cand, tot_coll = 0, 0
    colls_per_atom = zeros(Float64, cellcount)
    for cell in 1:cellcount
        Nc = Ncs[cell]
        (Nc < 2) && continue # Skip cells with 1 particle
        offset = offsets[cell]
        # Compute maximum speed
        speeds = zeros(Float64, Nc)
        for i in 1 : Nc
            atom = atom_lookup[i + offset - 1]
            speeds[i] = norm(view(velocities, :, atom))
        end
        max_speed = maximum(speeds)
        # Select appropriate number of pairs
        Mraw = F * (dt * σ / Vc) * Nc * (Nc - 1) * max_speed
        # Note: no explicit division by two in computing Mraw, because
        # |v_rel|max = 2 * max_speed, so it cancels out.
        Mcand = ceil(Int64, Mraw)
        Mcoll = 0
        prob_adjust = Mraw / Mcand # Adjustment for rounding
        prob_coeff = prob_adjust / (2 * max_speed)
        # Check and perform collisions
        for _ in 1:Mcand
            i1, i2 = samplepair(rng, Nc)
            atom1 = atom_lookup[i1 + offset - 1]
            atom2 = atom_lookup[i2 + offset - 1]
            # Get current velocities
            u1 = view(velocities, :, atom1)
            u2 = view(velocities, :, atom2)
            urel = norm(u2 - u1)
            if rand() < urel * prob_coeff # Collision probability
                # Compute new relative velocity
                ϕ = 2 * π * rand(rng)
                cosθ = 2 * rand(rng) - 1
                sinθ = sqrt(1 - cosθ^2)
                vrel = urel .* [sinθ * cos(ϕ), sinθ * sin(ϕ), cosθ]
                # Update stored velocities
                vcm = 0.5 * (u1 + u2)
                v1 = vcm + 0.5 * vrel;
                v2 = vcm - 0.5 * vrel;
                velocities[:, atom1] = v1;
                velocities[:, atom2] = v2;
                # Update cell's collision count and maximum speed
                speeds[i1] = norm(v1)
                speeds[i2] = norm(v2)
                max_speed = maximum(speeds)
                #max_speed = max(max_speed, speeds[i1], speeds[i2])
                prob_coeff = prob_adjust / (2 * max_speed)
                Mcoll += 1
            end
        end
        #max_colls_per_atom = max(max_colls_per_atom, Mcoll / Nc)
        colls_per_atom[cell] = Mcoll / Nc
        tot_cand += Mcand
        tot_coll += Mcoll
    end

    # Calculate new maximum timestep based on collision rate
    nonzero_colls_per_atom = colls_per_atom[(!iszero).(colls_per_atom)]
    if length(nonzero_colls_per_atom) > 2
        peak_colls_per_atom = percentile(nonzero_colls_per_atom, 50)
        min_coll_time = dt / (2 * peak_colls_per_atom)
        # If peak collisions per atom is zero, then dt is Infinity, and the
        # timestep will be limited by the other constraints (motion/trapping)
        dt = 0.1 * min_coll_time
    else
        # If too few cells had collisions, place no restriction on the timestep.
        dt = Inf
    end

    return dt, tot_cand, tot_coll
end

# Atom loss effects: high-energy particles, three-body recombination &
# background collisions.
function atom_loss_step!(cloud, m, ε, τ_bg, dt)
    positions, velocities = cloud.positions, cloud.velocities
    N₀=cloud.Nt
    N = N₀
    # Three-body recombination - compute probabilities
    recomb_prob = zeros(Float64, N₀)
    dV = prod(cloud.cellsize)
    K = (1e-30) * 1e-6 # Loss rate constant (m^6 / s)
    for cell in 1:cloud.cellcount
        Nc = cloud.cell_occupancies[cell]
        P = K * (cloud.F)^2 * Nc * (Nc - 1) / (dV^2) * dt
        # Store probability for all atoms in cell
        offset = cloud.cell_offsets[cell]
        for i in offset : offset + Nc - 1
            atom = cloud.atom_lookup[i]
            recomb_prob[atom] = P
        end
    end

    # Perform atom loss
    for i in N₀:-1:1
        vx, vy, vz = view(velocities, :, i)
        ke = 0.5 * m * (vx^2 + vy^2 + vz^2)
        evap_loss = (2*ke > ε) # Perfect loss model #TODO: use ke + pe, not 2*ke
        bg_loss = ( rand() < 1 - exp(-dt/τ_bg) ) # Background collisions
        three_body_loss = ( rand() < recomb_prob[i] )
        if #=evap_loss || bg_loss ||=# three_body_loss
            # Remove atom by replacing it with atom from the end
            velocities[:,i] = view(velocities, :, N)
            positions[:,i] = view(positions, :, N)
            N -= 1
        end
    end
    cloud.Nt = N
    return nothing
end

# Repopulate cloud by creating a duplicate of each
# particle, with the opposite velocity.
function duplicate!(cloud)
    N₀ = cloud.Nt
    N₁ = 2 * N₀
    positions = view(cloud.positions, :, 1:N₀)
    velocities = view(cloud.velocities, :, 1:N₀)
    if N₁ > size(cloud.positions, 2) # Buffer requires resizing
        cloud.positions = hcat(positions, positions)
        cloud.positions = hcat(velocities, -velocities)
        cloud.accels = zeros(Float64, 3, N₁)
    else # No resize required
        cloud.positions[:, N₀+1 : N₁] = positions
        cloud.velocities[:, N₀+1 : N₁] = -velocities
    end

    cloud.Nt = N₁
    cloud.F /= 2

    return nothing
end

# Dummy measure function if no measurement function provided
null_measure(_...) = nothing

# Evolve initial particle population for desired duration
function evolve(positions, velocities, accel, duration, σ, ω_max, m;
                trap_depth = (t) -> Inf, F = 1, Nc = 1, measure = null_measure,
                dt_modifier = 1, rng=MersenneTwister(), τ_bg = Inf)
    # Note: Nc is the target for average # atoms per cell.
    # Initialise cloud
    cloud = AtomCloud(positions, velocities, F)

    # Initial peak density (1e-5 limits max cell width)
    peak_density, _ = assign_cells!(cloud, 1e-5, Nc)
    # Proportionality constant relating timestep with max. acceleration
    motion_limit = 0.00005
    # Maximum timestep based on trap frequency
    trap_dt = 0.05 * 2π / ω_max
    dt = dt_modifier * trap_dt # Initial timestep

    # Track progress
    prog_detail = 10000
    progress = Progress(prog_detail, dt = 1, desc = "Simulation progress: ",
                color = :green, showspeed = false, enabled = true, barlen=50)
    
    # Iterate simulation
    t = 0 # Time
    iter_count = 0
    start_time = now()
    while t < duration
        # Collisionless motion
        motion_dt = verlet_step!(cloud, accel, t, dt, motion_limit)

        # Sort atoms by cell
        peak_free_path = 1 / (4 * peak_density * σ);
        peak_density, gridshape = assign_cells!(cloud, peak_free_path, Nc)

        # Perform collisions
        coll_dt, cand_count, coll_count = collision_step!(cloud, dt, σ, rng)

        atom_loss_step!(cloud, m, trap_depth(t), τ_bg, dt)
        
        if cloud.Nt < size(cloud.positions, 2) / 2
            duplicate!(cloud)
        end

        # Increment time and then update timestep
        t += dt
        dt = dt_modifier * min(coll_dt, trap_dt, motion_dt)

        # External measurements on system after one full iteration
        Nt = cloud.Nt
        measure(view(cloud.positions, :, 1:Nt), view(cloud.velocities, :, 1:Nt),
                cand_count, coll_count, t)

        #Update progress
        if (mod(iter_count, 100) == 0) || (t >= duration)
            progressvalues = [
                ("Virtual time", @sprintf("%.3g / %.3g s", t, duration)),
                ("Real time", now() - start_time),
                ("Iterations", iter_count),
                ("Coll. dt", @sprintf("%.3g", coll_dt)),
                ("Motion dt", @sprintf("%.3g", motion_dt)),
                ("Trap dt", @sprintf("%.3g", trap_dt)),
                ("Cell size", @sprintf("%.3g x %.3g x %.3g", cloud.cellsize...)),
                ("Grid shape", @sprintf("%i x %i x %i", gridshape...)),
                ("Non-empty cell count", cloud.nonempty_count),
                ("Peak density", peak_density),
                ("Np", cloud.Nt * cloud.F),
                ("Nt", cloud.Nt)
            ]
            if t >= duration
                finish!(progress, showvalues = progressvalues)
            else
                update!(progress, ceil(Int64, prog_detail * t / duration),
                        showvalues = progressvalues)
            end
        end

        iter_count += 1
    end

    # Return final positions and velocities
    return cloud
end