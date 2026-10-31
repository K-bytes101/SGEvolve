import re
import json
from typing import List, Dict, Optional

from population import Individual, PropertyRecord
from agents.base_agent import BaseAgent
from utils import replace_placeholders

SOLUTION_NUM_PER_RESPONSE = 3
MAX_TOKENS = int(32768 * 2)

class Generator(BaseAgent):
    """
    Generator class, used to generate initial individuals (cold start)
    Generate a group of initial population individuals by calling the large model
    """

    def __init__(self, config, reward_system, llm_requester):
        """
        Initialize Generator
        
        Args:
            config: Configuration
            llm_requester: LLM requester
        """
        # Load configuration
        super().__init__(config, reward_system, llm_requester)

        # Cold start task description
        self.task_context_config = config.get('task_context')
        if not self.task_context_config or not self.task_context_config.get('enabled', False):
            self.task_context_prompt = ""
        else:
            self.task_context_prompt = self.template_loader("generator/task_context.txt")
            
            # Task context
            replacements = {}
            replacements["MEDIUM_OR_FORMAT"] = self.task_context_config.get("medium_or_format", "")
            replacements["OPERATIONAL_BOUNDARY"] = self.task_context_config.get("operational_boundary", "")
            self.task_context_prompt = replace_placeholders(self.task_context_prompt, replacements=replacements)

        # Cold start output format specification
        self.cold_start_output_example = [
            {
                "id": 1,
                "solution": "According to the above instructions and problem description, the solution you generate.",
                "strength": "str",
                "weakness": "str",
                "evolving_direction": "str",
            },
            "..."
        ]
        if self.enable_properties:
            for prop in self.properties:
                name = prop.get("name")
                self.cold_start_output_example[0][name] = 0.0

        # Load template
        self.cold_start_template = config['prompt'].get('cold_start_template')  # Cold start template
        if config["evolution_type"] == "code":
            self.prompts = self.template_loader("generator/generation_template_code.txt")
        elif config["evolution_type"] == "modeling":
            self.prompts = self.template_loader("generator/generation_template_gurobi.txt")
        else:
            self.prompts = self.template_loader("generator/generation_template.txt")

        # Prompt initialization
        self._init_prompt()
        self._init_generation_prompt()

    
    def _init_generation_prompt(self):
        """
        Initialize specific prompts for the generation module
        """
        replacements = {}
        # Task context
        replacements["TASK_CONTEXT"] = self.task_context_prompt
        
        # Other extraction lines (placeholder key must match prompt_library templates)
        if self.enable_properties:
            other_extraction_lines = []
            for prop in self.properties:
                name = prop.get("name")
                other_extraction_lines.append(f'- "{name}": {prop.get('description')}')
            replacements["OTHER_REFLECTION_INFO"] = "\n".join(other_extraction_lines)

        # Cold start output format
        replacements["COLD_START_OUTPUT_EXAMPLE"] = json.dumps(self.cold_start_output_example, ensure_ascii=False, indent=2)

        # Cold start solution template
        cold_start_template = self.config['prompt'].get('cold_start_template', "")
        if cold_start_template:
            cold_start_template = cold_start_template.rstrip('\n')
        replacements['COLD_START_TEMPLATE'] = cold_start_template

        self.prompts = replace_placeholders(self.prompts, replacements=replacements)


    def generate(self, num_inds: int, island_id: int = 0) -> List[Individual]:
        """
        Generate a specified number of initial individuals (generate in batches to应对LLM上下文长度限制)
        
        Args:
            num_inds: Number of individuals to generate
            island_id: Island ID

        Returns:
            List[Individual]: List of generated initial individuals
        """
        all_individuals = []

        # Initialize prompt, keep {NUM_VARIANTS}
        prompt = self.prompts

        # Calculate how many batches are needed
        num_batches = (num_inds + SOLUTION_NUM_PER_RESPONSE - 1) // SOLUTION_NUM_PER_RESPONSE
        
        for batch_idx in range(num_batches):
            # Calculate the number of individuals to generate for the current batch
            if batch_idx == num_batches - 1:
                # Last batch: generate the remainder
                batch_size = num_inds - batch_idx * SOLUTION_NUM_PER_RESPONSE
            else:
                # Other batches: generate SOLUTION_NUM_PER_RESPONSE individuals
                batch_size = SOLUTION_NUM_PER_RESPONSE
            
            # Get the prompt for the current batch
            prompt_cp = replace_placeholders(prompt, {'NUM_VARIANTS': str(batch_size)})

            batch_ok = False
            for attempt in range(self.max_retries):
                response = self.llm_requester.get_response(
                    prompt_cp, model=self.model_name, max_tokens=MAX_TOKENS
                )
                if response is None:
                    if attempt < self.max_retries - 1:
                        print(
                            f"[Warning] Cold start batch {batch_idx + 1} LLM has no response, "
                            f"重试 {attempt + 1}/{self.max_retries}"
                        )
                    else:
                        print(
                            f"[Warning] Cold start batch {batch_idx + 1} LLM has no response, "
                            f"Retried {self.max_retries} times, skip this batch"
                        )
                    continue
                try:
                    batch_individuals = self._parse_response_to_individuals(
                        response,
                        island_id=island_id,
                        generation=0,
                    )
                    all_individuals.extend(batch_individuals)
                    batch_ok = True
                    break
                except Exception as e:
                    if attempt < self.max_retries - 1:
                        print(
                            f"[Warning] Cold start batch {batch_idx + 1} parsing failed, "
                            f"重试 {attempt + 1}/{self.max_retries}: {e}"
                        )
                    else:
                        print(
                            f"[Warning] Cold start batch {batch_idx + 1} parsing failed, "
                            f"Retried {self.max_retries} times: {e}"
                        )

            if not batch_ok:
                continue

            # If enough individuals have been generated, exit early
            if len(all_individuals) >= num_inds:
                break
        
        # Return the specified number of individuals (possibly because the LLM generated some extra)
        return all_individuals[:num_inds]


    def _parse_response_to_individuals(
        self, 
        response: str, 
        island_id: int = 0,  # Default island ID is 0
        generation: int = 0,
    ) -> List[Individual]:
        """
        Extract JSON code blocks from LLM response and convert to Individual object list
        
        Args:
            response: Original response string from LLM, contains ```json ... ``` string
            island_id: Island ID
            generation: Current generation
        
        Returns:
            List[Individual]: Parsed individual list
        """
        individuals = []
        
        # Use regular expression to extract ```json ... ``` code blocks
        pattern = r'```\s*json\s*\n(.*?)```'
        match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
        json_block = match.group(1) if match else None

        # Parse JSON string
        json_data = json.loads(json_block, strict=False)

        # Iterate over each JSON block
        for idx, block_data in enumerate(json_data):
            try:
                extraction_info = {}
                property_dict_new = {}

                # Verify necessary fields
                if not isinstance(block_data, dict):
                    print(f"[Warning] 跳过 JSON 块 {idx + 1}: 不是字典类型")
                    continue
                
                code = block_data.get("solution", "")
                if not code:
                    print(f"[Warning] 跳过 JSON 块 {idx + 1}: 缺少 solution 字段")
                    continue
                
                # Process diversity indicators, only used when single-objective optimization
                if self.enable_properties:
                    # Check if block_data contains all property fields
                    missing_properties = [
                        prop.get("name")
                        for prop in self.properties
                        if prop.get("name") not in block_data
                    ]
                    if missing_properties:
                        print(f"[Warning] Skip JSON block {idx + 1}: missing property fields: {missing_properties}")
                        continue

                    # Fill the values of each property field in block_data into property_dict_new
                    for prop in self.properties:
                        property_name = prop.get("name")
                        range = prop.get("range")
                        for_llm = prop.get("for_llm")
                        evaluator = prop.get("evaluator")

                        if property_name in block_data:
                            property_dict_new[property_name] = PropertyRecord(
                                score=float(block_data[property_name]),
                                range=range,
                                for_llm=for_llm,
                                evaluator=evaluator,
                            )
                        else:
                            property_dict_new[property_name] = PropertyRecord(
                                score=0,
                                range=range,
                                for_llm=for_llm,
                                evaluator=evaluator,
                            )

                # Fill in extraction_info
                for extraction_field in ['strength', 'weakness', 'evolving_direction']:
                    if extraction_field in block_data:
                        extraction_info[extraction_field] = [block_data[extraction_field]]
                    else:
                        print(f"[Warning] Cold start individual missing {extraction_field} field")
                        extraction_info[extraction_field] = []

                # Create Individual object
                individual = Individual(
                    island_id=island_id,
                    is_elite=False,
                    locked=False,
                    is_removed=False,
                    generation=generation,
                    parent_ids=[],
                    origin="init",
                    iteration_found=0,
                    code=code,
                    fitness=[],  # Initial fitness is empty, filled by evaluation later
                    property_dict=property_dict_new,
                    extraction_info=extraction_info,
                    eval_info={},
                    error_info={},
                )
                
                individuals.append(individual)
                print(f"  ✓ Successfully parsed JSON block {idx + 1}, id={block_data.get('id', 'N/A')}")
                
            except json.JSONDecodeError as e:
                print(f"[Warning] JSON block {idx + 1} parsing failed: {e}")
                continue
            except Exception as e:
                print(f"[Warning] Error processing JSON block {idx + 1}: {e}")
                continue
        
        print(f"  [Info] Island {island_id} successfully created {len(individuals)} Individual objects")
        return individuals


    def _add_diversity_guidance(self, prompt: str, gen_num: int, diversity_hints: Optional[List[str]]) -> str:
        """
        Add diversity guidance, ensure the generated individuals have diversity
        
        Args:
            prompt: Basic prompt
            gen_num: The current individual is the nth individual
            diversity_hints: Diversity hint list
        
        Returns:
            str: 添加了多样性指导的提示词
        """
        # TODO
        pass