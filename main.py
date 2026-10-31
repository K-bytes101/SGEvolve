if __name__ == "__main__":
    import os
    import yaml
    from copy import deepcopy
    import time
    from pathlib import Path
    import argparse

    from population import Population, Individual
    from agents.generator import Generator
    from utils.llm_requester import LLMRequester
    from agents.rewardsystem import RewardSystem
    from agents.extractor import Extractor
    from runner_core import run_evolution

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="HybridEvo - Evolutionary Code Optimization")
    parser.add_argument("--evolve", action="store_true", default=False,
                        help="Run evolution loop")
    parser.add_argument("--cold-start", action="store_true", default=False,
                        help="Execute cold start, generate initial population")
    parser.add_argument("--inject_init_solution", action="store_true", default=False,
                        help="Inject the initial solution into the population")
    parser.add_argument("--problem", type=str, default="sofa_moving",
                        help="Problem name (default: sofa_moving)")
    
    args = parser.parse_args()

    # Get configuration from command line arguments
    EVOLVE = args.evolve
    COLD_START = args.cold_start
    INJECT_INIT_SOLUTION = args.inject_init_solution and COLD_START
    CLEAR = COLD_START or INJECT_INIT_SOLUTION
    problem_name = args.problem

    print(f"Main process configuration: EVOLVE = {EVOLVE}, COLD_START = {COLD_START}, INJECT_INIT_SOLUTION = {INJECT_INIT_SOLUTION}, CLEAR = {CLEAR}, PROBLEM = {problem_name}")

    # Check if the problem directory and necessary files exist
    problem_dir = Path(__file__).parent / "problem" / problem_name
    if not problem_dir.exists():
        raise FileNotFoundError(
            f"Problem directory not found: {problem_dir}\n"
            f"Please ensure that the '{problem_name}' folder exists in the problem/ directory"
        )
    
    if not problem_dir.is_dir():
        raise NotADirectoryError(f"{problem_dir} is not a directory")

    # Check necessary files
    config_path = problem_dir / "config.yaml"
    evaluator_path = problem_dir / "evaluator.py"
    
    missing_files = []
    if not config_path.exists():
        missing_files.append("config.yaml")
    if not evaluator_path.exists():
        missing_files.append("evaluator.py")
    
    if missing_files:
        missing_list = "\n  - ".join(missing_files)
        raise FileNotFoundError(
            f"Problem directory '{problem_name}' is missing necessary files:\n  - {missing_list}\n\n"
            f"Please ensure that the problem directory contains the following files:\n"
            f"  - config.yaml (Problem configuration file)\n"
            f"  - evaluator.py (Evaluator file)"
        )

    # Load configuration
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    exp_num = os.getenv("EXP_NUM") or 1
    exp_num = int(exp_num)
    print(f"Experiment number: {exp_num}")

    # Initialize population
    db_url = f"sqlite:///./problem/{problem_name}/local_population_{exp_num}.db"
    pop = Population(config, db_url=db_url)

    # Initialize agents
    api_key = os.getenv("API_KEY")
    if api_key is None:
        raise Exception("API_KEY environment variable is not set")

    base_url = os.getenv("BASE_URL")
    if base_url is None:
        raise Exception("BASE_URL environment variable is not set")

    llm_requester = LLMRequester(api_key=api_key, base_url=base_url)
    reward_system = RewardSystem(config)

    generation_agent = Generator(config, reward_system, llm_requester)

    # Parameter initialization
    total_iterations = config["population"].get("iterations", 100)  # iteration budget
    pop_size = config["population"].get("population_size", 20)  # sub-population size
    island_num = config["population"].get("island_num", 1)  # island number
    print(f"Population configuration: \n"
          f"  Total iterations: {total_iterations}\n"
          f"  Population size: {pop_size}\n"
          f"  Island number: {island_num}\n")

    t0 = time.perf_counter()

    # Clear all individuals
    if CLEAR:
        pop.db.clear_all()

    # Cold start
    if COLD_START:
        print("Cold start...")
        for island_id in range(island_num):
            print(f"Island {island_id} cold start...")
            cold_start_num_inds = int(pop_size * 0.6)  # cold start generated individuals number
            cold_start_inds = generation_agent.generate(num_inds=cold_start_num_inds, island_id=island_id)

            # Evaluate all initialized individuals
            for index, ind in enumerate(cold_start_inds):
                print(f"  Evaluating cold start individual {index+1}...")
                fitness, eval_info, error_info = reward_system.evaluate(ind.code)
                
                # Print evaluation information
                print(f"    Fitness: {fitness}")
                print(f"    Evaluation info: {eval_info}")
                print(f"    Error info: {error_info}")
                
                cold_start_inds[index].fitness = fitness
                cold_start_inds[index].eval_info = eval_info
                cold_start_inds[index].error_info = error_info

            if not pop.init_population(cold_start_inds, island_id=island_id):
                raise Exception(f"[Error] Population initialization failed")

    # Initial solution
    if INJECT_INIT_SOLUTION:
        print("Injecting initial solutions...")
        init_solution_dir = Path(__file__).parent / config["init_solution_dir"]
        fake_parent = pop.db.get_by_id(1)

        init_solutions = []
        init_solution_files = os.listdir(init_solution_dir)
        for idx, init_solution_file in enumerate(init_solution_files):
            # Get the path of each initial solution
            init_solution_path = init_solution_dir / init_solution_file
            with open(init_solution_path, "r", encoding="utf-8") as f:
                init_solution = f.read()
            
            ind = ind = Individual(
                island_id=0,
                is_elite=False,
                locked=False,
                is_removed=False,
                generation=0,
                parent_ids=[],
                origin="init",
                iteration_found=0,
                code=init_solution,
                fitness=[],
                property_dict=deepcopy(fake_parent.property_dict),
                extraction_info=deepcopy(fake_parent.extraction_info),
                eval_info={},
                error_info={},
            )

            fitness, eval_info, error_info = reward_system.evaluate(ind.code)
            print(f"  Evaluating initial solution {idx+1}...")
            print(f"    Fitness: {fitness}")
            print(f"    Evaluation info: {eval_info}")
            print(f"    Error info: {error_info}")

            ind.fitness = fitness
            ind.eval_info = eval_info
            ind.error_info = error_info

            Extractor(config, reward_system, llm_requester, island_id=0, is_inject_init_solution=True).operate(ind, parents=[])
            init_solutions.append(ind)
        
        for island_id in range(island_num):
            print(f"Island {island_id} injecting initial solution...")
            for ind in init_solutions:
                ind_new = Individual(
                    island_id=island_id,
                    is_elite=False,
                    locked=False,
                    is_removed=False,
                    generation=0,
                    parent_ids=[],
                    origin="init",
                    iteration_found=0,
                    code=ind.code,
                    fitness=ind.fitness,
                    property_dict=deepcopy(ind.property_dict),
                    extraction_info=deepcopy(ind.extraction_info),
                    eval_info=ind.eval_info,
                    error_info=ind.error_info,
                )
                if not pop.add_individual(ind_new, island_id=island_id):
                    raise Exception(f"[Error] Island {island_id} initial solution injection failed")

    if EVOLVE:
        # Enter parallel evolving process
        batch_size = config["population"].get("batch_size", 3)
        run_evolution(
            config=config,
            db_url=db_url,
            total_iterations=total_iterations,  # total_iterations as the global total budget
            island_num=island_num,
            batch_size=batch_size,
        )
    elapsed = time.perf_counter() - t0

    # Print evolution results
    for idx in range(island_num):
        print(f"Island {idx} evolution results:")
        evolved_inds = pop.db.find(where=pop._build_where_conditions(island_id=idx, is_removed=False))
        evolved_inds = sorted(
            evolved_inds,
            key=lambda x: float(x.fitness[0]) if x.fitness and len(x.fitness) > 0 else float('-inf'),
        )
        for ind in evolved_inds:
            print(f"  Individual {ind.id} fitness: {ind.fitness[0]:.6f}, generation: {ind.generation}, is_elite: {ind.is_elite}, is_locked: {ind.locked}")
        print()

    print(f"Total time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
