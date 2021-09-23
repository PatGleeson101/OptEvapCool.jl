module OptEvapCool

export
kB, # Boltzmann constant
AtomSpecies, Rb87, # Atoms
uniform_cloud, boltzmann_velocities, # Atom initialisation
harmonic_boltzmann_positions,
GaussianBeam, HarmonicField, # Field initialisation
acceleration, potential,
no_evap, energy_evap, radius_evap, # Evaporation
GlobalSensor, measurer, measure, # Measurement and plotting
plot_energy, plot_temperature, plot_speed,
plot_collrate, plot_number,
savefig, savecsv, filetime, # File saving
SimulationConditions, # Setup
evolve # Perform the simulation

include("./conditions.jl")
include("./measure.jl")
include("./dsmc.jl")

end