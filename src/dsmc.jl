# DSMC

using StatsBase: samplepair, percentile
using LinearAlgebra: norm
using Random: MersenneTwister
using Printf: @sprintf
using ProgressMeter: Progress, update!, finish!
using Dates: now

# Storage of dynamic simulation data
mutable struct CloudBuffer
    positions :: Matrix{Float64}
    velocities :: Matrix{Float64}

    accels :: Matrix{Float64}
    potens :: Vector{Float64}
    survival_prob :: Vector{Float64}

    assignments :: Vector{Int64}
    cell_offsets :: Vector{Int64}
    cell_occupancies :: Vector{Int64}
    atom_lookup :: Vector{Int64}

    Nt :: Int64
    F :: Float64

    cellcount :: Int64
    nonempty_count :: Int64
    cellsize :: Vector{Float64}

    function CloudBuffer(conditions)
        # Atom buffers
        pos = copy(conditions.positions)
        vel = copy(conditions.velocities)
        Nt = size(pos, 2)
        acc = zeros(Float64, 3, Nt)
        pot = zeros(Float64, Nt)
        prob = zeros(Float64, Nt)
        assign = zeros(Int64, Nt)
        atoms = zeros(Int64, Nt)

        # Cell buffers (overwritten on first iteration)
        offsets = zeros(Int64, 1)
        occupancies = zeros(Int64, 1)

        # Dynamic cell info (updated on every iteration)
        cellsize = [0.0, 0.0, 0.0]
        nonempty_count = 1
        cellcount = 1

        # Return cloud
        return new(pos, vel, acc, pot, prob, assign, offsets, occupancies, atoms,
                   Nt, conditions.F, cellcount, nonempty_count, cellsize)
    end
end

# Velocty Verlet step + calculate motion-based limit on next timestep
function verlet_step!(cloud, conditions, t, dt)
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
    conditions.acceleration(positions, conditions.species, t + half_dt, accels)
    # Compute final velocities and positions, as well as new timestep
    for atom in 1:Nt
        ax, ay, az = view(accels, :, atom)
        velocities[1, atom] += dt * ax # Update velocity
        velocities[2, atom] += dt * ay
        velocities[3, atom] += dt * az
        vx, vy, vz = view(velocities, :, atom)
        positions[1, atom] += half_dt * vx # Update position
        positions[2, atom] += half_dt * vy
        positions[3, atom] += half_dt * vz
    end
end

# Assign particles to cells
function assign_cells!(cloud, peak_free_path, Nc, Vprev)
    N = cloud.Nt
    positions = view(cloud.positions, :, 1:N)

    # Corners of the bounding box around all atoms
    minx, miny, minz = Inf, Inf, Inf
    maxx, maxy, maxz = -Inf, -Inf, -Inf
    for atom in 1:N
        x, y, z = view(positions, :, atom)
        if x < minx; minx = x; elseif x > maxx; maxx = x; end;
        if y < miny; miny = y; elseif y > maxy; maxy = y; end;
        if z < minz; minz = z; elseif z > maxz; maxz = z; end;
    end

    minpos = [minx, miny, minz]
    maxpos = [maxx, maxy, maxz]

    # Cell dimensions
    V = min(Vprev, prod(maxpos - minpos)) # Volume estimate
    Vc = V * Nc / N
    ds = min(cbrt(Vc), 0.1 * peak_free_path)
    cellsize = [ds, ds, ds]
    cloud.cellsize = cellsize

    # Extend grid slightly beyond the furthest atoms, to
    # avoid edge atoms being placed outside the grid.
    lowerpos = minpos - (0.25 * cellsize)
    upperpos = maxpos + (0.25 * cellsize)
    gridshape = ceil.(Int64, (upperpos - lowerpos) ./ cellsize)
    cellcount = prod(gridshape)

    # Cell occupancies
    occupancy_dict = Dict{Int64, Int64}()

    # Assign atoms to cells and count the number of atoms in each cell.
    assignments = cloud.assignments
    xcount, ycount, _ = gridshape
    lx, ly, lz = lowerpos # Cache lower coordinates
    for atom in 1:N
        x = ceil(Int64, (positions[1, atom] - lx) / ds)
        y = ceil(Int64, (positions[2, atom] - ly) / ds)
        z = ceil(Int64, (positions[3, atom] - lz) / ds)
        cell = x + xcount * ( (y - 1) + ycount * (z - 1) )
        if haskey(occupancy_dict, cell)
            occupancy_dict[cell] += 1
        else
            occupancy_dict[cell] = 1
        end
        assignments[atom] = cell
    end

    cellcount = length(occupancy_dict) # ONLY NONEMPTY CELLS

    # Ensure storage buffers are large enough.
    if cellcount > length(cloud.cell_offsets)
        cloud.cell_offsets = zeros(Int64, cellcount * 2)
        cloud.cell_occupancies = zeros(Int64, cellcount * 2)
    end
    cell_offsets = view(cloud.cell_offsets, 1:cellcount)
    cell_occupancies = view(cloud.cell_occupancies, 1:cellcount)
    fill!(cell_offsets, 0)
    fill!(cell_occupancies, 0)

    # Map cell numbers to nonempty cell index, and
    # compute storage offset for each cell
    index_dict = Dict{Int64, Int64}()
    i = 1
    offset = 1
    for (cell, Nc) in occupancy_dict
        if Nc > 0
            index_dict[cell] = i
            cell_offsets[i] = offset
            cell_occupancies[i] = Nc
            offset += Nc
            i += 1
        end
    end

    # Create lookup index of atoms in each cell.
    # This temporarily modifies the cell offsets.
    atom_lookup = cloud.atom_lookup
    for atom in 1:N
        cell = assignments[atom]
        cell_index = index_dict[cell]
        atom_lookup[ cell_offsets[cell_index] ] = atom
        cell_offsets[cell_index] += 1
    end

    # Restore cell offsets
    for cell in 1:cellcount
        Nc = cell_occupancies[cell]
        cell_offsets[cell] -= Nc
    end

    cloud.nonempty_count = cellcount
    cloud.cellcount = cellcount

    # Compute peak density.
    peak_density = maximum(cell_occupancies) / Vc
    #percentile(cell_occupancies, 97) / Vc

    Vnew = cellcount * Vc # Total volume of non-empty cells

    return peak_density, gridshape, Vnew
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
    Vc = prod(cloud.cellsize) # Cell volume
    tot_cand, tot_coll = 0, 0
    for cell in 1:cellcount
        Nc = Ncs[cell]
        (Nc < 2) && continue # Skip cells with 1 particle
        offset = offsets[cell]
        # Compute maximum speed
        speeds = zeros(Float64, Nc)
        for i in 1 : Nc
            atom = atom_lookup[i + offset - 1]
            vx, vy, vz = view(velocities, :, atom)
            speeds[i] = sqrt(vx^2 + vy^2 + vz^2)
        end
        max_speed = maximum(speeds)
        # Select appropriate number of pairs
        Mraw = F * (dt * σ / Vc) * Nc * (Nc - 1) * max_speed
        # Note: |v_rel|max = 2 * max_speed, so expression slightly different
        # to theory in report.
        Mcand = ceil(Int64, Mraw)
        Mcoll = 0 # Total collisions in cell
        prob_adjust = Mraw / Mcand # Adjustment for rounding
        Pcoeff = prob_adjust / (2 * max_speed)
        # Check and perform collisions
        for _ in 1:Mcand
            i1, i2 = samplepair(rng, Nc)
            atom1 = atom_lookup[i1 + offset - 1]
            atom2 = atom_lookup[i2 + offset - 1]
            # Get current velocities
            u1x, u1y, u1z = view(velocities, :, atom1)
            u2x, u2y, u2z = view(velocities, :, atom2)
            urel = sqrt((u1x - u2x)^2 + (u1y - u2y)^2 + (u1z - u2z)^2)
            Pij = urel * Pcoeff
            if rand(rng) < Pij # Collision prob.
                # Compute new relative velocity
                ϕ = 2 * π * rand(rng)
                cosθ = 2 * rand(rng) - 1
                sinθ = sqrt(1 - cosθ^2)
                vrx = urel * sinθ * cos(ϕ)
                vry = urel * sinθ * sin(ϕ)
                vrz = urel * cosθ
                # Update stored velocities
                vcmx = 0.5 * (u1x + u2x)
                vcmy = 0.5 * (u1y + u2y)
                vcmz = 0.5 * (u1z + u2z)
                v1x = vcmx + 0.5 * vrx;
                v1y = vcmy + 0.5 * vry;
                v1z = vcmz + 0.5 * vrz;
                v2x = vcmx - 0.5 * vrx;
                v2y = vcmy - 0.5 * vry;
                v2z = vcmz - 0.5 * vrz;
                velocities[1, atom1] = v1x
                velocities[2, atom1] = v1y
                velocities[3, atom1] = v1z
                velocities[1, atom2] = v2x
                velocities[2, atom2] = v2y
                velocities[3, atom2] = v2z
                # Update cell's collision count and maximum speed
                speed1 = sqrt(v1x^2+ v1y^2+v1z^2)
                speed2 = sqrt(v2x^2+ v2y^2+ v2z^2)
                speeds[i1] = speed1
                speeds[i2] = speed2
                max_speed = max(max_speed, speed1, speed2) #maximum(speeds)
                Pcoeff = prob_adjust / (2 * max_speed)
                Mcoll += 1
            end
        end
        tot_cand += Mcand
        tot_coll += Mcoll
    end

    # Calculate collision-based timestep limit
    Γₐ = 2 * tot_coll / (cloud.Nt * dt) # Mean collision rate per atom
    dt = 0.05/Γₐ

    return dt, tot_cand, tot_coll
end

# Atom loss effects: high-energy particles, three-body recombination &
# background collisions.
function atom_loss!(cloud, conditions, t, dt, rng=MersenneTwister())
    τ_bg, K, evap = conditions.τ_bg, conditions.threebodyloss, conditions.evaporate

    N₀, F = cloud.Nt, cloud.F
    dV = prod(cloud.cellsize)
    positions = view(cloud.positions, :, 1:N₀)
    velocities = view(cloud.velocities, :, 1:N₀)

    # Evaporation LOSS probability for each atom (becomes survival later)
    p_survive = view(cloud.survival_prob, 1:N₀)
    evap(positions, velocities, conditions, t, p_survive)

    p_background = 1 - (dt / τ_bg) # Background survival probability (const.)
    for cell in 1:cloud.cellcount
        Nc = cloud.cell_occupancies[cell]
        offset = cloud.cell_offsets[cell]
        # Three-body-loss survival probability (constant within cell)
        p_threebody = 1 - (K * F^2 * Nc * (Nc - 1) / (dV^2) * dt)
        for i in offset : offset + Nc - 1
            # Update survival likelihood to reflect 3-body + background loss
            atom = cloud.atom_lookup[i]
            p_survive[atom] = (1.0 - p_survive[atom]) * p_background * p_threebody
        end
    end

    # Perform atom loss
    N = N₀
    for atom in N₀:-1:1
        if rand(rng) > p_survive[atom]
            # Remove atom by replacing it with atom from the end
            velocities[:,atom] .= view(velocities, :, N)
            positions[:,atom] .= view(positions, :, N)
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
        @warn "Cloud duplication triggered above half capacity."
        cloud.positions = hcat(positions, positions)
        cloud.positions = hcat(velocities, velocities)
        # NOTE: switched from reflected velocities to identical velocities.
        accels = view(cloud.accels, :, 1:N₀)
        cloud.accels = hcat(accels, accels)
    else # No resize required
        cloud.positions[:, N₀+1 : N₁] .= positions
        cloud.velocities[:, N₀+1 : N₁] .= velocities
    end

    cloud.Nt = N₁
    cloud.F /= 2

    return nothing
end

# Evolve initial conditions for desired duration
function evolve(conditions, duration; Nc = 1, max_dt = Inf,
    rng=MersenneTwister(), measure = null, meas_dt = 1e-4)
    #= Arguments
    - conditions: a SimulationConditions
    - duration: in virtual time
    - Nc: target average number of atoms per cell.
    - max_dt: upper bound on timestep.
    - rng: allows reproducibility if desired.
    =#

    # Initialise dynamic storage
    cloud = CloudBuffer(conditions)

    σ = conditions.species.σ
    max_dt = time_parametrize(max_dt)
    
    # Initial values
    peak_density, _, V = assign_cells!(cloud, 1e-5, Nc, Inf)
    dt = min(max_dt(0), 1e-9) # Begin with very short timestep

    # Track progress
    prog_detail = 10000
    progress = Progress(prog_detail, dt = 1, desc = "Simulation progress: ",
                color = :green, showspeed = false, enabled = true, barlen=25)
    
    accum_cand = 0 # Accumulated since previous measurement
    accum_coll = 0
    accum_n0 = 0
    accum_V = 0
    meas_iter = 0
    meas_t = -Inf # Previous measurement time
    
    # Iterate simulation
    t = 0 # Virtual time
    iter_count = 0
    start_time = now()
    while t < duration
        # Collisionless motion
        verlet_step!(cloud, conditions, t, dt)

        # Sort atoms by cell
        peak_free_path = 1 / (4 * peak_density * σ);
        peak_density, gridshape, V = assign_cells!(cloud, peak_free_path, Nc, V)

        # Perform collisions
        coll_dt, cand_count, coll_count = collision_step!(cloud, dt, σ, rng)

        atom_loss!(cloud, conditions, t, dt, rng)
        
        if cloud.Nt == 0
            @warn "Trap empty"
            return cloud
        end

        while cloud.Nt < size(cloud.positions, 2) / 2
            duplicate!(cloud)
        end

        # Increment time and then update timestep
        t += dt
        dt = min(coll_dt, max_dt(t))

        # Record for measurements
        accum_cand += cand_count
        accum_coll += coll_count
        accum_n0 += peak_density
        accum_V += V

        iter_count += 1

        # External measurements on system at specified intervals
        if t >= meas_t + meas_dt
            elapsed_iter = iter_count - meas_iter
            n0 = accum_n0 / elapsed_iter
            V_avg = accum_V / elapsed_iter
            measure(cloud, conditions, accum_cand, accum_coll, t, n0, V_avg)
            accum_cand, accum_coll = 0, 0
            accum_n0, accum_V = 0, 0
            meas_t = t
            meas_iter = iter_count
        end
        
        #Update progress
        if (mod(iter_count, 100) == 0) || (t >= duration)
            progressvalues = [
                ("Virtual time", @sprintf("%.3g / %.3g s", t, duration)),
                ("Real time", now() - start_time),
                ("Iterations", iter_count),
                ("Coll. dt", @sprintf("%.3g", coll_dt)),
                ("Max. dt", @sprintf("%.3g", max_dt(t))),
                ("dt", @sprintf("%.3g", dt)),
                ("Cell size", @sprintf("%.3g x %.3g x %.3g", cloud.cellsize...)),
                ("Grid shape", @sprintf("%i x %i x %i", gridshape...)),
                ("Non-empty cell count", cloud.nonempty_count),
                ("Peak density", cloud.F * peak_density),
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
    end

    # Return final positions and velocities
    return cloud
end