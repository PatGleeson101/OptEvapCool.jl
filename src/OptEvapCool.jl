module OptEvapCool

export
kB, # Boltzmann constant
AtomSpecies, Rb87, kappa, # Atoms
uniform_cloud, boltzmann_velocities, # Atom initialisation
harmonic_boltzmann_positions,
GaussianBeam, HarmonicField, UniformField, # Field initialisation
gravity, acceleration, potential,
no_evap, energy_evap, radius_evap, # Evaporation
exponential_ramp, linear_ramp,
GlobalSensor, measurer, measure, # Measurement and plotting
plot_energy, plot_temperature, plot_speed,
plot_collrate, plot_number, plot_density,
plot_psd,
equilibrium_collrate, equilibrium_speeds, # Theoretical predictions
harmonic_theory, purdue_theory,
savefig, savecsv, filetime, loadsensor, # File saving/loading
SimulationConditions, # Setup
evolve # Perform the simulation

include("./conditions.jl")
include("./measure.jl")
include("./dsmc.jl")
include("./theory.jl")

end