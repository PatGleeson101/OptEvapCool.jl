# Test of DSMC with Harmonic potential
# TODO: disable collisions in this test.

# using Pkg
# Pkg.add("BenchmarkTools")
# Pkg.add("LinearAlgebra)
# Pkg.add("Calculus")

# using BenchmarkTools
# using Calculus
# using LinearAlgebra

using Plots
using Random
include("../src/OptEvapCool.jl")
#using .OptEvapCool

Random.seed!(1) # Set random generator seed

# Measurement storage
const time = zeros(0)
const pe = zeros(0)
const ke = zeros(0)
const avg_dist = zeros(0)
const temp = zeros(0)

function test()
    # Simulation parameters
    dt = 1e-4 # timestep
    N = convert(UInt64, 1e5) # initial number of atoms
    v_th = 0.01
    gas_size = 10e-6 # approximate size of gas
    duration = 0.1
    m = 1.44e-25 # Atomic mass (kg)

    # Rough thermal velocity estimate for Rb-87 at 1 microKelvin: 0.01 m/s
    positions, velocities = OptEvapCool.init_uniform(N, v_th, gas_size)
    ω_x = 2π * 150;
    ω_y = 2π * 150;
    ω_z = 2π * 15;
    accel, potential = OptEvapCool.harmonic(ω_x, ω_y, ω_z)

    measure = OptEvapCool.record_all(time, ke, pe, avg_dist, temp,
                                     potential, positions, m)

    # Run evolution
    OptEvapCool.evolve!(positions, velocities, accel, duration, dt, measure)
end

test()

# Plot energy 
energy_plt = plot(time, [ke, pe, ke + pe],
     title = "Total energy over time",
     label = ["Kinetic" "Potential" "Total"],
     yformatter = :scientific)
xlabel!("Time (s)")
ylabel!("Energy (J)")

# Plot average distance from initial positions
distance_plt = plot(time, avg_dist,
     title = "Average distance from start position",
     legend = false)
xlabel!("Time (s)")
ylabel!("Distance (m)")

# Plot average temperature over time
temperature_plt = plot(time, temp,
     title = "Average temperature over time",
     legend = false)
xlabel!("Time (s)")
ylabel!("Temperature (K)")

display(energy_plt)
display(distance_plt)
display(temperature_plt)