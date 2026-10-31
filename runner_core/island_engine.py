import os
import random
from typing import List
from population.individual import Individual
from utils import format_obj_list


class IslandEngine:
    def __init__(self, island_id: int, config, db_url, stop_event=None):
        self.island_id = island_id
        self.config = config
        self.stop_event = stop_event
        self.objective_list = format_obj_list(config["prompt"].get("objectives", []))

        # ===== Lazy import (avoid spawn problem) =====
        from population import Population
        from agents.mutator import Mutator
        from agents.crossover import Crossover
        from agents.extractor import Extractor
        from utils.llm_requester import LLMRequester
        from agents.rewardsystem import RewardSystem

        # ===== Initialize components =====
        self.pop = Population(config, db_url=db_url)

        api_key = os.getenv("API_KEY")
        if api_key is None:
            raise RuntimeError("API_KEY is not set")
        
        base_url = os.getenv("BASE_URL")
        if base_url is None:
            raise RuntimeError("BASE_URL is not set")

        llm_requester = LLMRequester(api_key=api_key, base_url=base_url)
        self.reward_system = RewardSystem(config)

        self.mutation_agent = Mutator(config, self.reward_system, llm_requester, island_id)
        self.crossover_agent = Crossover(config, self.reward_system, llm_requester, island_id)
        self.extractor_agent = Extractor(config, self.reward_system, llm_requester, island_id)

        # Unlock population individuals, avoid sampling error caused by previous evolution interruption
        self.pop.unlock_island(self.island_id)

    def _should_stop_on_ground_truth(self, fitness) -> bool:
        """Return True when evaluation.stop_on_ground_truth_match is True and all objectives fitness reaches/exceeds benchmark"""
        ev = self.config.get("evaluation") or {}
        if not ev.get("stop_on_ground_truth_match", False):
            return False
        if not fitness or len(fitness) != len(self.objective_list):
            return False
        tol = float(ev.get("ground_truth_match_tol", 1e-4))
        for fv, obj in zip(fitness, self.objective_list):
            bench = obj.get("benchmark")
            if bench is None:
                return False
            bench = float(bench)
            v = float(fv)
            if abs(v - bench) > tol:
                return False
        return True

    def unlock_parents(self, parents: List[Individual]):
        """
        Unlock sampled parents
        """
        if not parents:
            return

        for parent in parents:
            if not self.pop.unlock_sampled_ind(parent.id):
                print(f"Warning: Unlock parent {parent.id} failed")

    def evolve_once(self, curr_iteration: int):
        """
        Single step evolution

        Args:
            curr_iteration: Current iteration number
        """
        if self.stop_event is not None and self.stop_event.is_set():
            return

        print(f"Island {self.island_id}, Iteration {curr_iteration} starts...")
        oper_prob = random.random()
        warning_flag = False
        warning_message = ""
        try:
            # ========== Mutation ==========
            if oper_prob < self.config["mutation_rate"]:
                warning_message += f"[Warning] Island {self.island_id}, Mutation stage: \n"
                # Sampling
                parents = self.pop.mutation_sampling(island_id=self.island_id)
                if parents is None:
                    warning_message += f"   Sampling failed, skip this iteration\n"
                    print(warning_message)
                    self.unlock_parents(parents)
                    return
                # Get child individual
                child = self.mutation_agent.operate(parents[0])

            # ========== Crossover ==========
            elif oper_prob < self.config["mutation_rate"] + self.config["crossover_rate"]:
                warning_message += f"[Warning] Island {self.island_id}, Crossover stage: \n"
                parents = self.pop.crossover_sampling(island_id=self.island_id)
                if parents is None:
                    warning_message += f"   Sampling failed, skip this iteration\n"
                    print(warning_message)
                    self.unlock_parents(parents)
                    return
                child = self.crossover_agent.operate(parents)

            # Get extraction information
            if not child:
                warning_flag = True
                warning_message += f"   No child individual obtained, skip this iteration\n"
            else:
                child.iteration_found = curr_iteration
                extraction_payload = self.extractor_agent.operate(child, parents)
                if extraction_payload is None:
                    warning_flag = True
                    warning_message += f"   No extraction data obtained\n"
                
                # Add to population
                self.pop.add_individual(child, self.island_id)
                if self.stop_event is not None and self._should_stop_on_ground_truth(child.fitness):
                    self.stop_event.set()
                    print(
                        f"[Island {self.island_id}] Early stop: all objectives matched benchmarks "
                        f"(stop_on_ground_truth_match)."
                    )
                    self.reward_system.close()

            # Unlock sampled parents
            self.unlock_parents(parents)

            # Print warning information
            if warning_flag:
                print(warning_message)

        except Exception as e:
            self.unlock_parents(parents)
            print(f"[Island {self.island_id}] Error: {e}")
