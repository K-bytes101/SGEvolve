from typing import Optional, Tuple

from agents.base_agent import BaseAgent
from population import Individual
from utils import replace_placeholders

class Mutator(BaseAgent):
    def __init__(self, config, reward_system, llm_requester, island_id: int):
        super().__init__(config, reward_system, llm_requester)
        self.island_id = island_id
        self.prompts = self.template_loader("mutator/mutation_template.txt")
        self._init_prompt()


    def _get_mutation_prompt(self, parent: Individual):
        """
        Get prompt for mutation

        Args:
            parent (Individual): Parent individual

        Returns:
            str: Prompt for mutation
        """
        
        # Prepare placeholder replacement dictionary
        replacements = {}

        # Evaluation data
        replacements["FITNESS"] = self._get_fitness_prompt(parent.fitness)  # Fitness value
        replacements["EVAL_DETAILS"] = self._dict2prompt(parent.eval_info)  # Evaluation information
        if parent.error_info:
            replacements["EVAL_LOGS"] = self._dict2prompt(parent.error_info)  # Error information
        else:
            replacements["EVAL_LOGS"] = "Evaluation completed successfully."

        # Code
        replacements["EVOL_ENTITY"] = parent.code
        # Evolution direction
        evolving_direction = parent.extraction_info.get("evolving_direction")
        replacements["EVOL_DIRECTION"] = self._list2prompt(evolving_direction)

        prompt = replace_placeholders(self.prompts, replacements)

        return prompt


    def operate(self, parent: Individual) -> Optional[Individual]:
        """
        Mutation operation

        Args:
            parent: Parent individual
        Returns:
            child: Child individual obtained by mutation
        """
        # Get new individual
        prompt = self._get_mutation_prompt(parent)
        response = self.llm_requester.get_response(prompt, model=self.model_name, max_tokens=self.max_tokens)
        code = self.extract_code_from_llm(response=response, language=self.coding_language)

        # New individual is empty, return None
        if code is None:
            return None

        fitness, eval_info, error_info = self.reward_system.evaluate(code)

        # Print Mutation stage information
        print(f"Island {self.island_id}, Mutation stage information: \n")
        print(f"   Parent information: parent id: {parent.id}")
        print(f"    fitness: {parent.fitness}")
        print(f"    eval_info: {parent.eval_info}")
        print(f"    error_info: {parent.error_info}\n")

        print(f"   Child information:")
        print(f"    fitness: {fitness}")
        print(f"    eval_info: {eval_info}")
        print(f"    error_info: {error_info}\n")

        # Create child individual
        child = Individual(
            island_id=self.island_id,
            is_elite=False,
            locked=False,
            is_removed=False,
            parent_ids=[parent.id],
            generation=parent.generation + 1,
            origin="mutation",
            iteration_found=0,
            code=code,
            fitness=fitness,
            property_dict=self.property_dict if self.enable_properties else {},
            extraction_info={},
            eval_info=eval_info,
            error_info=error_info,
        )

        return child