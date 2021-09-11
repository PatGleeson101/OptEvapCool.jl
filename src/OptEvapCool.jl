module OptEvapCool

export
kB, # Boltzmann constant
harmonic_field, exponential_ramp, # Cloud and field initialisation methods
uniform_cube_cloud, boltzmann_velocities, harmonic_boltzmann_positions,
evolve, # Perform the simulation
avg_kinetic_energy, avg_potential_energy, record # Measurement functions

include("setup.jl")
include("dsmc.jl")
include("measure.jl")

end