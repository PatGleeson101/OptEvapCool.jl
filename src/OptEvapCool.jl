module OptEvapCool

export
kB, # Boltzmann constant
AtomSpecies, Rb87, kappa, # Atoms
uniform_cloud, boltzmann_velocities, # Atom initialisation
harmonic_boltzmann_positions,
GaussianBeam, HarmonicField, UniformField, # Field initialisation
gravity, acceleration, potential,
no_evap, energy_evap, radius_evap, # Evaporation
exponential_ramp,
GlobalSensor, measurer, measure, # Measurement and plotting
plot_energy, plot_temperature, plot_speed,
plot_collrate, plot_number,
harmonic_eq_collrate, harmonic_eq_speeds, # Theoretical predictions
savefig, savecsv, filetime, loadsensor, # File saving/loading
SimulationConditions, # Setup
evolve # Perform the simulation

include("./conditions.jl")
include("./measure.jl")
include("./dsmc.jl")

end