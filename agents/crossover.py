from typing import List, Dict, Optional

from agents.base_agent import BaseAgent
from population import Individual
from utils import replace_placeholders

class Crossover(BaseAgent):
    def __init__(self, config, reward_system, llm_requester, island_id: int):
        super().__init__(config, reward_system, llm_requester)
        self.island_id = island_id
        self.prompts = self.template_loader("crossover/crossover_template.txt")
        self._init_prompt()


    def _get_crossover_prompt(self, parents: List[Individual]):
        """
        Get crossover prompt
        """
        parent_a = parents[0]
        parent_b = parents[1]

        # Prepare placeholder replacement dictionary
        replacements = {}
        # Parent data
        replacements["PARENT_A_FITNESS"] = self._get_fitness_prompt(parent_a.fitness)
        replacements["PARENT_A_EVAL_DETAILS"] = self._dict2prompt(parent_a.eval_info)
        if parent_a.error_info:
            replacements["PARENT_A_EVAL_LOGS"] = self._dict2prompt(parent_a.error_info)  # Error information
        else:
            replacements["PARENT_A_EVAL_LOGS"] = "Evaluation completed successfully."

        replacements["PARENT_B_FITNESS"] = self._get_fitness_prompt(parent_b.fitness)
        replacements["PARENT_B_EVAL_DETAILS"] = self._dict2prompt(parent_b.eval_info)
        if parent_a.error_info:
            replacements["PARENT_B_EVAL_LOGS"] = self._dict2prompt(parent_b.error_info)  # Error information
        else:
            replacements["PARENT_B_EVAL_LOGS"] = "Evaluation completed successfully."

        # Code
        replacements["PARENT_A_EVOL_ENTITY"] = parent_a.code
        replacements["PARENT_B_EVOL_ENTITY"] = parent_b.code
        # Strength and weakness
        replacements["PARENT_A_STRENGTH"] = self._list2prompt(parent_a.extraction_info["strength"])
        replacements["PARENT_A_WEAKNESS"] = self._list2prompt(parent_a.extraction_info["weakness"])
        replacements["PARENT_B_STRENGTH"] = self._list2prompt(parent_b.extraction_info["strength"])
        replacements["PARENT_B_WEAKNESS"] = self._list2prompt(parent_b.extraction_info["weakness"])

        prompt = replace_placeholders(self.prompts, replacements)

        return prompt

    def operate(self, parents: List[Individual]) -> Optional[Individual]:
        """
        Crossover operation
        """
        # Get child individual
        prompt = self._get_crossover_prompt(parents)
        response = self.llm_requester.get_response(prompt, model=self.model_name, max_tokens=self.max_tokens)
        code = self.extract_code_from_llm(response=response, language=self.coding_language)

        if code is None:
            return None

        fitness, eval_info, error_info = self.reward_system.evaluate(code)

        # Print Crossover stage information
        print(f"Island {self.island_id}, Crossover stage information: \n")
        print(f"  Parent information: parent id: {parents[0].id} and {parents[1].id}")
        for parent in parents:
            print(f"    fitness: {parent.fitness}")
            print(f"    eval_info: {parent.eval_info}")
            print(f"    error_info: {parent.error_info}\n")

        print(f"  Child information:")
        print(f"    fitness: {fitness}")
        print(f"    eval_info: {eval_info}")
        print(f"    error_info: {error_info}\n\n")

        # Create child individual
        child = Individual(
            island_id=self.island_id,
            is_elite=False,
            locked=False,
            is_removed=False,
            parent_ids=[parents[0].id, parents[1].id],
            generation=max(parents[0].generation, parents[1].generation) + 1,
            origin="crossover",
            iteration_found=0,
            code=code,
            fitness=fitness,
            property_dict=self.property_dict if self.enable_properties else {},
            extraction_info={},
            eval_info=eval_info,
            error_info=error_info,
        )

        return child