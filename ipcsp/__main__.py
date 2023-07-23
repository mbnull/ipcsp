"""
This is the code to reproduce Table 1 and assess performance of a D-Wave quantum annealer for CSP.
The latter by default uses simulated annealing implementation on a classical computer provided by
D-Wave and a quantum annealer can be accessed after a registration with a single parameter change.

Some tests can be omitted by toggling 'test' value to False in the settings dictionary below.
Some parameters of the configuration spaces can be changed there as well.

Integer programming solver Gurobi is used to solve the following structures:
1. SrTiO3 for different supercells (perovskite structure)
2. Y2O3 (bixbyite structure)
3. Y2Ti2O7 (pyrochlore structure)
4. MgAl2O4 (spinel structure)
5. Ca3Al2Si3O12 (garnet structure)

Quantum experiments are presented for SrO, SrTiO3, ZnS, ZrO2.
"""

from time import time
import os
import shutil

from tabulate import tabulate
import pandas as pd

from ipcsp import root_dir
from integer_program import Allocate
from matrix_generator import Phase
from ase.calculators.gulp import GULP
import ase.io
from copy import deepcopy

'''
 The settings dictionary lists the predictions to run and parameters of the configuration spaces for
 integer programs. It is divided in two parts: to reproduce Table 1 in the paper and quantum experiments
 
 Common parameters:
   'test' -- True if the test is chosen to run
   'multiple' -- the number of repeats of the unit cell per direction, essentially, we are predicting multiple
     copies of the structure simultaneously (mostly relevant for SrTiO3)
  'group' -- space group of an allocation. Positions that are equal under symmetry, will have the same atoms
  'grid' -- the parameter g equal to the number of points per the side of a unit cell
  'top' -- the number of the lowest energy solutions that will be computed by Gurobi
    If top is 1, then only the global optimum will be considered. The values of top > 1 should be used with caution.
    Equivalent allocations (but different solutions of the integer program) with the same energy likely will be produced,
    thus, the number should be larger than the required number of different solutions. Further, the current version of
    Gurobi can occasionally produce solutions violating constraints. This is a bug, which should be fixed 
    in future versions of Gurobi based on forum discussions. We deal with this issue by simply filtering out incorrect 
    solutions for the time being.
   
 Quantum annealing specific parameters: 
   'at_dwave' -- True will connect to a D-Wave quantum annealer and use your computational budget (register first)
   'at_dwave' -- False will rely on the local simulated annealing
   'num_reads' -- the number of solutions that will be sampled using annealing
    'annealing_time' -- how long the quantum annealing will take per sample. Slower "readouts" can occasionally 
      lead to better results occasionally
    'infinity_placement' and 'infinity_orbit' are parameters gamma and mu defined in the paper to energetically 
       penalise allocations that have incorrect stoichiometry and have two atoms on top of each other
       
Note, Gurobi is called first in the quantum section as well as a shortcut to generate coefficients 
of the the integer program corresponding to the periodic lattice atom allocation problem.
It is written into model.lp. The file lp_to_bqm.py contains tools to convert this model into a QUBO problem
that can be submitted to the quantum annealer. We don't do a local minimisation step here as the structures
are relatively simple. 
'''

settings = {

    # The first part is done using Gurobi on a local machine.
    # Every line corresponds to a row in Table 1.

    # perovskite structure of SrTiO
    'SrTiO3_1': {'test': True, 'multiple': 1, 'group': 1, 'top': 1, 'grid': 4},  # group is 221, sub 195
    'SrTiO3_2': {'test': True, 'multiple': 2, 'group': 195, 'top': 1, 'grid': 8},  # group is 221, sub 195
    'SrTiO3_3': {'test': True, 'multiple': 3, 'group': 221, 'top': 1, 'grid': 6},  # group is 221, sub 195
    'SrTiO3_4': {'test': True, 'multiple': 3, 'group': 200, 'top': 1, 'grid': 6},  # group is 221, sub 195
    'SrTiO3_5': {'test': True, 'multiple': 3, 'group': 195, 'top': 1, 'grid': 6},  # group is 221, sub 195
}


def process_results(lib, results, ions_count, test_name, printing=False):
    # path hack here
    os.mkdir(os.path.join("..", "results", test_name))
    calc = GULP(keywords='single', library=os.path.join(".", lib))

    # stash the allocations for future
    results_ip = deepcopy(results)

    # Compute the number of atoms
    N_atoms = 0
    for k, v in ions_count.items():
        N_atoms += v

    init = [0] * len(results)
    final = [0] * len(results)
    best_val = 0
    best_idx = 0
    # ase.io.write("best_ipcsp.vasp", results[0])
    # ase.io.write(os.path.join("..", "results", test_name, "ip_optimum.vasp"), results[0])
    print("Processing and locally optimising solutions from the integer program\n")
    for idx, cryst in enumerate(results):
        if len(cryst.arrays['positions']) == N_atoms:
            cryst.calc = calc
            init[idx] = cryst.get_potential_energy()
            # print("Initial:", init[idx])
        else:
            print("GULP received a bad solution. Gurobi's implementation of pooling occasionally provides solutions "
                  "that do not satisfy constraints. It should be corrected in future versions of the solver.")

    calc.set(keywords='opti conjugate conp diff comp c6')
    prev_energy = -1000000
    for idx, cryst in enumerate(results):
        if init[idx] < -0.00001:
            if init[idx] - prev_energy > 0.000001:
                prev_energy = init[idx]
                opt = calc.get_optimizer(cryst)
                try:
                    opt.run(fmax=0.05)
                    final[idx] = cryst.get_potential_energy()
                except ValueError:
                    print("One of the relaxations failed using initial energy instead")
                    final[idx] = init[idx]

                if final[idx] < best_val:
                    best_idx = idx
                    best_val = final[idx]
                # print("Final:", final[idx])
                # input()
            # print("Energy initial: ", cryst.get_potential_energy(), " final: ", final)

    count = 1
    with open(os.path.join("..", "results", test_name, "energies.txt"), "w+") as f:
        for i in range(len(results)):
            if final[i] != 0:
                print(f"Solution{count}: ", "Energy initial: ", init[i], " final: ", final[i])
                print(f"Solution{count}: ", "Energy initial: ", init[i], " final: ", final[i], file=f)
                # if len(results) > 1:
                #    # ase.io.write(f'solution{count}.vasp', results[i])
                ase.io.write(os.path.join("..", "results", test_name, f'solution{count}_lattice.vasp'), results_ip[i])
                ase.io.write(os.path.join("..", "results", test_name, f'solution{count}_minimised.vasp'), results[i])
                count += 1

    cryst = results[best_idx]
    print("The lowest found energy is ", best_val, "eV")
    # print("The energy per ion is ", best_val/N_atoms, "eV")
    # ase.io.write(os.path.join("..", "results", test_name, "minimal_energy_structure.vasp"), cryst)
    # print("Rerunning GULP, so that gulp.gout would have optimised structure")
    # opt = calc.get_optimizer(cryst)
    # opt.run(fmax=0.05)
    # cryst.get_potential_energy()
    if printing:
        print("Paused, the files can be copied")
        input()

    return best_val


def get_cif_energies(filename, library, format='cif'):
    filedir = root_dir / 'structures/'
    # Path hacks again
    cryst = ase.io.read(os.path.join(".", filedir / filename), format=format, parallel=False)
    calc = GULP(keywords='conp', library=library)
    calc.set(keywords='opti conjugate conp diff comp c6')
    opt = calc.get_optimizer(cryst)
    opt.run(fmax=0.05)
    energy = cryst.get_potential_energy()

    print("The energy of", filename, "is equal to", energy, "eV")

    return energy


def benchmark():
    # Preparing a folder with results

    shutil.rmtree(os.path.join("..", "results"), ignore_errors=True)
    os.mkdir(os.path.join("..", "results"))

    '''
    
    Single test selector
    
    for key in settings.keys():
        settings[key]['test'] = False

    settings['quantum_ZnS']['test'] = True
    
    #'''

    df_summary = pd.DataFrame(columns=['name', 'grid', 'group', 'best_E', 'expected_E', 'time'])

    for i in range(1, 6):
        if settings[f'SrTiO3_{i}']['test']:
            print("\n\n\n========== Predicting SrTiO3 (perovskite structure) ==========")
            print(settings[f'SrTiO3_{i}'])

            SrTiO = Phase('SrTiO')

            multiple = settings[f'SrTiO3_{i}']['multiple']

            ions_count = {'O': 3 * multiple ** 3, 'Sr': 1 * multiple ** 3, 'Ti': 1 * multiple ** 3}

            start = time()
            allocation = Allocate(ions_count, grid_size=settings[f'SrTiO3_{i}']['grid'], cell_size=3.9 * multiple,
                                  phase=SrTiO)

            # The correct symmetry group is 221, supergroup of 195
            results, runtime, _ = allocation.optimize_cube_symmetry_ase(group=settings[f'SrTiO3_{i}']['group'],
                                                                        PoolSolutions=settings[f'SrTiO3_{i}']['top'],
                                                                        TimeLimit=0)

            best_energy = process_results(lib=SrTiO.filedir / 'SrTiO/buck.lib', results=results,
                                          ions_count=ions_count, test_name=f'SrTiO_{i}')

            energy = get_cif_energies(filename='SrTiO3.cif', library=SrTiO.filedir / 'SrTiO/buck.lib')
            if multiple > 1:
                energy = energy * multiple ** 3
                print("For the given multiple it is equal to ", energy, "eV")

            end = time()
            print('It took ', end='')
            print(" %s seconds including IP and data generation" % (end - start))

            df_summary = df_summary.append({'name': f'SrTiO3_{i}', 'grid': settings[f'SrTiO3_{i}']['grid'],
                                            'group': settings[f'SrTiO3_{i}']['group'], 'best_E': best_energy,
                                            'expected_E': energy, 'time': runtime}, ignore_index=True)
            with open(os.path.join("..", "results", "summary.txt"), "w+") as f:
                print("Non-heuristic optimisation using Gurobi with subsequent local minimisation (test equivalent to Table 1 of the paper):", file=f)
                print(tabulate(df_summary, headers=["Test name", "Discretisation g", "Space group","Best energy (eV)", "Target energy (eV)", "IP solution time (sec)"],
                       tablefmt='github', showindex=False), file=f)


    with open(os.path.join("..", "results", "summary.txt"), "a") as f:
        print("\n\n\n\n\n Quantum annealing for the periodic lattice atom allocation.\n", file=f)
        print(tabulate(df_summary, headers=["Test name", "D-Wave", "Best energy (eV)", "Target energy (eV)"],
                       tablefmt='github', showindex=False), file=f)


if __name__ == "__main__":
    benchmark()
