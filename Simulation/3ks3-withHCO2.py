import torch
import numpy as np
from ase.io import read, write
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.langevin import Langevin
from ase import units
from orb_models.forcefield import atomic_system, pretrained
from ase.calculators.calculator import Calculator, all_properties
from ase.build import molecule, make_supercell
from ase.md import MDLogger
from orb_models.forcefield.calculator import ORBCalculator
from ase.constraints import FixAtoms


def setup_device():
    """Set up and return the appropriate compute device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    return device


class DistanceCapCalculator(Calculator):
    """Calculator wrapper that adds distance cap forces after the main calculation."""
    
    def __init__(self, base_calculator, atom_i, atom_j, r_max=5.0, k=5.0, debug=False):
        super().__init__()
        self.base_calculator = base_calculator
        self.atom_i = atom_i
        self.atom_j = atom_j
        self.r_max = r_max
        self.k = k
        self.debug = debug
        self.implemented_properties = self.base_calculator.implemented_properties

    def calculate(self, atoms=None, properties=['energy', 'forces'], system_changes=all_properties):
        # First get the base calculator results
        self.base_calculator.calculate(atoms=atoms, properties=properties, system_changes=system_changes)
        
        # Copy results
        self.results = self.base_calculator.results.copy()
        
        # Now we can safely modify the forces from the results
        forces = self.results['forces']  # Get forces from results
        pos = atoms.get_positions()
        
        # Vector from j to i
        rij_vec = pos[self.atom_i] - pos[self.atom_j]
        dist_ij = np.linalg.norm(rij_vec)

        # Only apply force if distance exceeds r_max
        if dist_ij > self.r_max and dist_ij > 1e-12:
            e_ij = rij_vec / dist_ij
            # Hookean force for r > r_max
            F_mag = -self.k * (dist_ij - self.r_max)  # negative => pulls them closer
            F_i = F_mag * e_ij
            F_j = -F_i

            # Capture old forces for debug
            old_f_i = forces[self.atom_i].copy()
            old_f_j = forces[self.atom_j].copy()

            # Update forces
            forces[self.atom_i] += F_i
            forces[self.atom_j] += F_j

            if self.debug:
                print(f"[DEBUG] Distance cap engaged for atoms i={self.atom_i}, j={self.atom_j}.")
                print(f"        Current distance = {dist_ij:.3f} Å > r_max={self.r_max:.3f} Å.")
                print(f"        Force magnitude  = {abs(F_mag):.3f} eV/Å.")
                print(f"        Old force on i: {old_f_i}, new force on i: {forces[self.atom_i]}")
                print(f"        Old force on j: {old_f_j}, new force on j: {forces[self.atom_j]}")


def run_md_simulation(
    input_file: str = "3ks3-withHCO2noGOL.xyz",
    cell_size: float = 100,
    temperature_K: float = 310,
    timestep: float = 0.5 * units.fs,
    friction: float = 0.01 / units.fs,
    total_steps: int = 1000000,
    traj_interval: int = 20,
    log_interval: int = 100,
):
    """Run molecular dynamics simulation with specified parameters.
    
    Args:
        input_file: Path to input XYZ file
        cell_size: Size of cubic simulation cell
        temperature_K: Temperature in Kelvin
        timestep: MD timestep
        friction: Langevin friction coefficient
        total_steps: Total number of MD steps
        traj_interval: Interval for trajectory writing
        log_interval: Interval for log writing
    """
    # Set up device
    device = setup_device()
    
    # Read in the system from file and set the cell size and pbc
    atoms = read(input_file, index=-1)
    atoms.set_cell([cell_size,cell_size,cell_size])
    atoms.set_pbc([False] * 3)

    # Read frozen atom indices
    with open('frozenatoms.dat', 'r') as f:
        frozen_indices = [int(x) for x in f.read().split()]

    # Apply constraints to fix specified atoms
    constraint = FixAtoms(indices=frozen_indices)
    atoms.set_constraint(constraint)

    # Set up the base calculator
    base_calc = ORBCalculator(
        model=pretrained.orb_d3_v2(),
        device=device
    )
    
    # Wrap the base calculator with DistanceCapCalculator
    constrained_calc = DistanceCapCalculator(
        base_calculator=base_calc,
        atom_i=4054,
        atom_j=5499,
        r_max=5.0,  # 5 Angstrom max distance
        k=5.0,
        debug=True
    )
    
    # Set the constrained calculator
    atoms.calc = constrained_calc

    # Set the initial velocities
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K)

    # Zero velocities of the fixed atoms
    velocities = atoms.get_velocities()
    velocities[frozen_indices] = 0.0
    atoms.set_velocities(velocities)

    # Set the dynamics
    dyn = Langevin(atoms, timestep, temperature_K=temperature_K, friction=friction)

    # Define output functions and attach to dynamics
    dyn.attach(
        lambda: write('3ks3-withHCO2noGOL-MD.xyz', atoms, format='xyz',append=True), 
        interval=traj_interval
    )
    dyn.attach(MDLogger(dyn, atoms, "md_nvt.log"), interval=log_interval)

    # Run the dynamics
    dyn.run(steps=total_steps)


def main():
    """Main entry point for the script."""
    run_md_simulation()


if __name__ == "__main__":
    main()
