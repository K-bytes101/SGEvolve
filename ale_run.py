# ALE-Bench 评测入口

if __name__ == "__main__":
    import os
    import yaml
    import argparse
    import ale_bench
    from copy import deepcopy
    from pathlib import Path

    from population import Population
    from agents.generator import Generator
    from utils.llm_requester import LLMRequester
    from agents.extractor import Extractor
    from agents.rewardsystem import RewardSystem
    from runner_core import run_evolution
    from population.individual import Individual
    # Parse the command line arguments
    parser = argparse.ArgumentParser(description="HybridEvo - Evolutionary Code Optimization")
    parser.add_argument("--evolve", action="store_true", default=False,
                        help="Run evolutionary loop")
    parser.add_argument("--cold-start", action="store_true", default=False,
                        help="Execute cold start, generate initial population")
    parser.add_argument("--inject_init_solution", action="store_true", default=False,
                        help="Inject the initial solution into the population")
    parser.add_argument("--result", action="store_true", default=False,
                        help="Print the results directly")
    parser.add_argument(
        "--problem-ids",
        nargs="+",
        default=None,
        metavar="ID",
        help="Problem IDs (e.g. ahc008 ahc011 ahc012). If omitted, use built-in default list.",
    )
    args = parser.parse_args()

    # Get the configuration from the command line arguments
    RESULT = args.result  # Only show the results
    EVOLVE = args.evolve and not RESULT
    COLD_START = args.cold_start and not RESULT
    INJECT_INIT_SOLUTION = args.inject_init_solution and COLD_START and not RESULT  # Only inject the initial solution when cold starting
    CLEAR = COLD_START and not RESULT  # Clear the database when cold starting
    
    problem_name = "ALE-Bench"
    default_problem_ids = [
        "ahc008", "ahc011", "ahc015", "ahc016", "ahc024", "ahc025",
        "ahc026", "ahc027", "ahc039", "ahc046",
    ]
    problem_ids = list(args.problem_ids) if args.problem_ids is not None else default_problem_ids

    print(
        f"Main Process Configuration: EVOLVE = {EVOLVE}, COLD_START = {COLD_START}, INJECT_INIT_SOLUTION = {INJECT_INIT_SOLUTION}, "
        f"CLEAR = {CLEAR}, PROBLEM = {problem_name}, PROBLEM_IDS = {problem_ids}"
    )

    # Initialize the environment variables
    api_key = os.getenv("API_KEY")
    if api_key is None:
        raise Exception("API_KEY environment variable is not set")
    base_url = os.getenv("BASE_URL")
    if not base_url:
        raise Exception("BASE_URL environment variable is not set")
    if os.getenv("ALE_BENCH_DATA") is None:
        os.environ["ALE_BENCH_DATA"] = "./problem/ALE-Bench/dataset/ALE-Bench"
        print(f"ALE_BENCH_DATA environment variable is set to {os.environ['ALE_BENCH_DATA']}")
    if os.getenv("ALE_BENCH_SHARED") is None:
        raise Exception("ALE_BENCH_SHARED environment variable is not set")

    if os.getenv("TMPDIR") is None:
        raise Exception("TMPDIR environment variable is not set")
    if os.getenv("ALE_BENCH_CACHE") is None:
        raise Exception("ALE_BENCH_CACHE environment variable is not set")

    # Check if the problem directory and necessary files exist
    problem_dir = Path(__file__).parent / "problem" / problem_name
    if not problem_dir.exists():
        raise FileNotFoundError(
            f"Problem directory not found: {problem_dir}\n"
            f"Please ensure that the '{problem_name}' folder exists in the problem/ directory"
        )
    
    if not problem_dir.is_dir():
        raise NotADirectoryError(f"{problem_dir} is not a directory")

    # Check the necessary files
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
            f"Problem directory '{problem_name}' is missing required files:\n  - {missing_list}\n\n"
            f"Please ensure that the problem directory contains the following files:\n"
            f"  - config.yaml (Problem configuration file)\n"
            f"  - evaluator.py (Evaluator file)"
        )

    # Load the configuration
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Initialize the LLM API interface
    llm_requester = LLMRequester(api_key=api_key, base_url=base_url)

    # Initialize the parameters
    total_iterations = config["population"].get("iterations", 100)  # 迭代预算
    pop_size = config["population"].get("population_size", 20)  # 子种群大小
    island_num = config["population"].get("island_num", 1)  # 岛屿数量
    # island_num = 1
    print(f"Population configuration: \n"
          f"  Iteration budget: {total_iterations}\n"
          f"  Subpopulation size: {pop_size}\n"
          f"  The number of islands: {island_num}\n")

    baseline_self_refine_level = "16x"

    # Loop
    for problem_id in problem_ids:
        # Initialize the population
        db_url = f"sqlite:///./problem/{problem_name}/local_population_{problem_id}.db"
        pop = Population(config, db_url=db_url)

        # Only show the results
        if RESULT:
            # private eval: open a new independent session, avoid being contaminated by the evolution subprocess
            print(f"Opening a fresh session for private eval (problem: {problem_id})...")
            private_session = ale_bench.start(
                problem_id,
                lite_version=True,
                num_workers=8,
                run_visualization_server=False,
            )
            try:
                all_inds = pop.db.find(where=pop._build_where_conditions(is_removed=False))
                all_inds = sorted(
                    all_inds,
                    key=lambda x: float(x.fitness[0]) if x.fitness and len(x.fitness) > 0 else float('-inf'),
                    reverse=True,
                )
                for ind in all_inds:
                    private_result, final_rank, final_performance = private_session.private_eval(ind.code, code_language="cpp20")
                    print(f"Individual {ind.id} fitness: {ind.fitness[0]:.6f}, private score: {private_result.overall_absolute_score}, rank: {final_rank}, performance: {final_performance}")
            finally:
                private_session.close()

        print("--------------------------------")
        print(f"Problem: {problem_id}")
        print("--------------------------------")
        # Load the problem
        print(f"Loading problem {problem_id}...")
        session = ale_bench.start(
            problem_id,
            lite_version=True,
            num_workers=16,
            run_visualization_server=False,
        )

        problem = session.problem
        config["session"] = session
        config["prompt"]["problem_description"] = problem.statement

        # Initialize Agents
        reward_system = RewardSystem(config)
        generation_agent = Generator(config, reward_system, llm_requester)

        # Clear all individuals
        if CLEAR:
            pop.db.clear_all()

        # Cold start
        if COLD_START:
            print("Cold start...")
            for island_id in range(island_num):
                print(f"Island {island_id} cold start...")
                if INJECT_INIT_SOLUTION:
                    cold_start_num_inds = int(pop_size * 0.6)
                else:
                    cold_start_num_inds = int(pop_size * 0.6) - 1
                # cold_start_num_inds = 1
                cold_start_inds = generation_agent.generate(num_inds=cold_start_num_inds, island_id=island_id)

                # Evaluate all the initial individuals
                for index, ind in enumerate(cold_start_inds):
                    print(f"  Evaluating cold start individual {index+1}...")
                    fitness, eval_info, error_info = reward_system.evaluate(ind.code)
                    
                    # Print the evaluation information
                    print(f"    Fitness: {fitness}")
                    print(f"    Evaluation info: {eval_info}")
                    print(f"    Error info: {error_info}")
                    
                    cold_start_inds[index].fitness = fitness
                    cold_start_inds[index].eval_info = eval_info
                    cold_start_inds[index].error_info = error_info

                if not pop.init_population(cold_start_inds, island_id=island_id):
                    raise Exception(f"[Error] Population initialization failed")

        # Inject the initial solution
        if INJECT_INIT_SOLUTION:
            print("Injecting initial solution...")
            config["initial_program"] = f"problem/ALE-Bench/ALE_Agent_baseline/self-refine_{baseline_self_refine_level}/{problem_id}_initial_program.cpp"
            init_solution_path = Path(__file__).parent / config["initial_program"]
            # Sample a temporary parent, get the property_dict object
            fake_parent = pop.db.get_by_id(1)
            with open(init_solution_path, 'r', encoding='utf-8') as f:
                init_solution = f.read()
            
            ind = Individual(
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
            print(f"    Fitness: {fitness}")
            print(f"    Evaluation info: {eval_info}")
            print(f"    Error info: {error_info}")

            ind.fitness = fitness
            ind.eval_info = eval_info
            ind.error_info = error_info

            Extractor(config, reward_system, llm_requester, island_id=0, is_inject_init_solution=True).operate(ind, parents=[])

            for island_id in range(island_num):
                print(f"Island {island_id} injecting initial solution...")
                ind_new = Individual(
                    island_id=island_id,
                    is_elite=False,
                    locked=False,
                    is_removed=False,
                    generation=0,
                    parent_ids=[],
                    origin="init",
                    iteration_found=0,
                    code=init_solution,
                    fitness=fitness,
                    property_dict=deepcopy(ind.property_dict),
                    extraction_info=deepcopy(ind.extraction_info),
                    eval_info=eval_info,
                    error_info=error_info,
                )
                if not pop.add_individual(ind_new, island_id=island_id):
                    raise Exception(f"[Error] Population initialization failed")

        if EVOLVE:
            # Enter the parallel evolving process
            batch_size = config["population"].get("batch_size", 3)
            run_evolution(
                config=config,
                db_url=db_url,
                total_iterations=total_iterations,  # total_iterations 作为全局总预算
                island_num=island_num,
                batch_size=batch_size,
            )

        # Recycle the RewardSystem resources
        reward_system.close()

        # Close the session used in the cold-start / inject phase (the evolution subprocess has already closed it)
        try:
            session.close()
        except Exception:
            pass

        # Print the evolution results
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
