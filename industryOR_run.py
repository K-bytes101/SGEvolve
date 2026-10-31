if __name__ == "__main__":
    import os
    import json
    import yaml
    from copy import deepcopy
    import time
    from pathlib import Path
    import argparse

    from population import Population, Individual
    from agents.generator import Generator
    from utils.llm_requester import LLMRequester
    from agents.rewardsystem import RewardSystem
    from runner_core import run_evolution
    from datasets import load_from_disk

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="HybridEvo - Evolutionary Code Optimization")
    parser.add_argument("--evolve", action="store_true", default=False,
                        help="Run evolution loop")
    parser.add_argument("--cold_start", action="store_true", default=False,
                        help="Execute cold start, generate initial population")
    parser.add_argument("--modeling_mode", action="store_true", default=True,
                        help="Use Gurobi modeling mode (default)")
    
    args = parser.parse_args()

    # Get configuration from command line arguments
    EVOLVE = args.evolve
    COLD_START = args.cold_start
    CLEAR = COLD_START
    MODELING_MODE = args.modeling_mode

    print(f"Main process configuration: EVOLVE = {EVOLVE}, COLD_START = {COLD_START}, CLEAR = {CLEAR}, MODELING_MODE = {MODELING_MODE}, PROBLEM = IndustryOR")

    # Check if the problem directory and necessary files exist
    problem_dir = Path(__file__).parent / "problem" / "IndustryOR"
    if not problem_dir.exists():
        raise FileNotFoundError(
            f"Problem directory not found: {problem_dir}\n"
            f"Please ensure that the 'IndustryOR' folder exists in the problem/ directory"
        )
    
    if not problem_dir.is_dir():
        raise NotADirectoryError(f"{problem_dir} is not a directory")

    # Configuration & evaluator path
    config_path = problem_dir / "modeling_mode" / "config.yaml"
    evaluator_path = problem_dir / "modeling_mode" / "evaluator.py"

    # Load configuration
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # LLM interface
    api_key = os.getenv("API_KEY")
    if api_key is None:
        raise Exception("API_KEY environment variable is not set")
    base_url = os.getenv("BASE_URL")
    if base_url is None:
        raise Exception("BASE_URL environment variable is not set")
    llm_requester = LLMRequester(api_key=api_key, base_url=base_url)

    # Population parameters
    total_iterations = config["population"].get("iterations", 100)  # iteration budget
    pop_size = config["population"].get("population_size", 20)  # sub-population size
    island_num = config["population"].get("island_num", 1)  # island number
    print(f"Population configuration: \n"
          f"  Total iterations: {total_iterations}\n"
          f"  Population size: {pop_size}\n"
          f"  Island number: {island_num}\n")

    # Load data
    dataset = load_from_disk("problem/IndustryOR/dataset")["test"]

    score = 0  # Easy: 1 point; Medium: 2 points; Hard: 4 points; 200 points total
    results = {}
    
    # Task id
    task_to_run = [i for i in range(len(dataset))]
    dataset_run = dataset.select(task_to_run)
    for row in dataset_run:
        task_id = row["id"]
        difficulty = row["difficulty"]  # Easy: 40 tasks; Medium: 40 tasks; Hard: 20 tasks
        config["prompt"]["problem_description"] = row["en_question"]
        ground_truth = float(row["en_answer"])
        config["answer"] = ground_truth

        print("=" * 30 + f" Task id: {task_id}, Difficulty: {difficulty} " + "=" * 30)
        print(f"Task description: {row['en_question']}")
        
        # Initialize population
        db_url = f"sqlite:///./problem/IndustryOR/modeling_mode/local_population_id_{task_id}.db"
        pop = Population(config, db_url=db_url)

        # Reward system
        reward_system = RewardSystem(config)
    
        generation_agent = Generator(config, reward_system, llm_requester)

        # Clear all individuals
        if CLEAR:
            pop.db.clear_all()

        # Cold start
        early_stop_flag = False
        if COLD_START:
            print("  Cold start...")
            
            for island_id in range(island_num):
                print(f"  Island {island_id} cold start...")
                cold_start_num_inds = 3
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

                    if fitness[0] <= 1e-6:
                        early_stop_flag = True

                if not pop.init_population(cold_start_inds, island_id=island_id):
                    raise Exception(f"[Error] Population initialization failed")
                
                if early_stop_flag:
                    break
        # Cold start end
        reward_system.close()

        if early_stop_flag:
            results[task_id] = {
                "id": task_id,
                "difficulty": difficulty,
                "correct": True,
            }
            if difficulty == "Easy":
                score += 1
            elif difficulty == "Medium":
                score += 2
            elif difficulty == "Hard":
                score += 4
            print(f"Early stop on difficulty {difficulty} at task {task_id}")
            continue
 
        t0 = time.perf_counter()
        if EVOLVE:
            # Enter parallel evolving process
            batch_size = 1
            run_evolution(
                config=config,
                db_url=db_url,
                total_iterations=total_iterations,  # total_iterations as the global total budget
                island_num=island_num,
                batch_size=batch_size,
            )
        elapsed = time.perf_counter() - t0

        # Score
        right_answer = False
        for island_id in range(island_num):
            print(f"Island {island_id} evolution results:")
            evolved_inds = pop.db.find(where=pop._build_where_conditions(island_id=island_id, is_removed=False))
            for ind in evolved_inds:
                if ind.fitness[0] <= 1e-6:
                    right_answer = True
                    break
            if right_answer:
                break

        results[task_id] = {
            "id": task_id,
            "difficulty": difficulty,
            "correct": bool(right_answer),
        }

        if right_answer:
            if difficulty == "Easy":
                score += 1
            elif difficulty == "Medium":
                score += 2
            elif difficulty == "Hard":
                score += 4

        print(f"Task id: {task_id}, Difficulty: {difficulty}, Total time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")

    results["score"] = score
    results_path = problem_dir / "industryOR_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4, default=str)
    print(f"Results (score={score}) saved to {results_path}")
