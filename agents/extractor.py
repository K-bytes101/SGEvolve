import json
from typing import List, Dict, Optional

from agents.base_agent import BaseAgent
from population.individual import Individual
from utils import replace_placeholders

class Extractor(BaseAgent):
    def __init__(self, config, reward_system, llm_requester, island_id: int, is_inject_init_solution: bool = False):
        super().__init__(config, reward_system, llm_requester)
        self.island_id = island_id
        self.is_inject_init_solution = is_inject_init_solution
        # Select different build prompt paths based on whether properties are enabled
        if self.enable_properties:
            if self.is_inject_init_solution:
                self.prompts = self.template_loader("extractor/extraction_init_solution_with_properties_template.txt")
            else:
                self.prompts = self.template_loader("extractor/extraction_with_properties_template.txt")
        else:
            self.prompts = self.template_loader("extractor/extraction_template.txt")

        # Output format specification
        self.output_example = {
            "strength": ["strength 1", "..."],
            "weakness": ["weakness 1", "..."],
            "evolving_direction": ["direction 1", "..."],
        }
        if self.enable_properties:  # Additional diversity indicators
            for prop in self.properties:
                name = prop.get("name")
                self.output_example[name] = 0.0

        if not self.is_inject_init_solution:
            self.parent_info_template = self.template_loader("extractor/parent_info_template.txt")

        # Initialize prompt
        self._init_prompt()
        self._init_extraction_prompt()  # Initialize extractor module specific prompt


    def _init_extraction_prompt(self):
        """
        Initialize extractor module specific prompt
        """
        replacements = {}
        # Other extraction lines (placeholder key must match prompt_library templates)
        if self.enable_properties:
            other_extraction_lines = []
            for prop in self.properties:
                name = prop.get("name")
                other_extraction_lines.append(f'- "{name}": {prop.get("description")}')
            
            replacements["OTHER_REFLECTION_INFO"] = "\n".join(other_extraction_lines)
        
        # Output format
        replacements["OUTPUT_EXAMPLE"] = json.dumps(self.output_example, ensure_ascii=False, indent=2)

        self.prompts = replace_placeholders(self.prompts, replacements=replacements)

        # Parent information template
        if not self.is_inject_init_solution:
            self.parent_info_template = replace_placeholders(self.parent_info_template, replacements=replacements)


    def operate(self, child: Individual, parents: List[Individual] | Individual) -> Optional:
        """
        Extraction operation

        Args:
            child (Individual): Child individual
            parents (List[Individual] | Individual): Parent individual list or single parent individual

        Returns:
            extraction_data: Formatted extraction data
        """
        if not isinstance(parents, list):
            # If parents is not a list, convert it to a list
            parents = [parents]

        prompt = self._get_extraction_prompt(child, parents)
        response = self.llm_requester.get_response(prompt, model=self.model_name, max_tokens=self.max_tokens)
        extraction_data = self.extract_json_from_llm(response)

        if extraction_data is None:
            return None

        # Record extraction information
        for extraction_field in ['strength', 'weakness', 'evolving_direction']:
            if extraction_field in extraction_data:
                child.extraction_info[extraction_field] = extraction_data[extraction_field]
            else:
                print(f"[Warning] Extraction data missing {extraction_field} field")
                child.extraction_info[extraction_field] = []

        for prop_name in self.property_names:
            if prop_name in extraction_data:
                child.property_dict[prop_name].score = extraction_data[prop_name]
            else:
                print(f"[Warning] Extraction data missing {prop_name} field")
                child.property_dict[prop_name].score = 0

        return extraction_data

    def _get_extraction_prompt(self, child: Individual, parents: List[Individual] | Individual) -> str:
        """
        Get extraction prompt
        """
        replacements = {}

        # Prepare parent context information
        if not self.is_inject_init_solution:
            parental_context = []
            for idx, parent in enumerate(parents):
                replace_tmp = {}
                replace_tmp["IDX"] = idx + 1
                replace_tmp["FITNESS"] = self._get_fitness_prompt(parent.fitness)  # Fitness value
                replace_tmp["EVAL_DETAILS"] = self._dict2prompt(parent.eval_info)  # Evaluation information
                if parent.error_info:
                    replace_tmp["EVAL_LOGS"] = self._dict2prompt(parent.error_info)  # Error information
                else:
                    replace_tmp["EVAL_LOGS"] = "Evaluation completed successfully."

                # Language
                replace_tmp["LANGUAGE"] = self.coding_language

                # Code
                replace_tmp["EVOL_ENTITY"] = parent.code
                # Single parent information
                parent_info_prompt = replace_placeholders(self.parent_info_template, replacements=replace_tmp)
                parental_context.append(parent_info_prompt)
            replacements["PARENTAL_CONTEXT"] = "\n".join(parental_context)

        # Child information
        replacements["CHILD_FITNESS"] = self._get_fitness_prompt(child.fitness)
        replacements["CHILD_EVAL_DETAILS"] = self._dict2prompt(child.eval_info)
        if child.error_info:
            replacements["CHILD_EVAL_LOGS"] = self._dict2prompt(child.error_info)
        else:
            replacements["CHILD_EVAL_LOGS"] = "Evaluation completed successfully."
        replacements["CHILD_EVOL_ENTITY"] = child.code

        prompt = replace_placeholders(self.prompts, replacements=replacements)
        return prompt
