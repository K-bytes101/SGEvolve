import re
import json
from pathlib import Path

from typing import List, Dict, Optional
from population import Population, PropertyRecord
from utils import replace_placeholders, format_obj_list


class BaseAgent:
    def __init__(self, config, reward_system, llm_requester):
        self.config = config
        self.have_ground_truth = config.get("evaluation", {}).get("have_ground_truth", False)
        self.reward_system = reward_system  # Feedback system
        self.llm_requester = llm_requester  # API call
        self.model_name = config["model"]
        self.max_tokens = config.get('max_tokens', 4096)
        self.coding_language = config.get("coding_language")
        self.max_retries = config.get("max_retries", 3)
        
        # Get optimization objective list
        self.objective_list = format_obj_list(config['prompt'].get('objectives', []))
        if not self.objective_list:
            raise ValueError("Objectives are not found, please check the objectives configuration.")

        # Property configuration
        self.property_config = config.get("property_config")
        if len(self.objective_list) < 2 and not self.property_config:
            raise ValueError("Property configuration is not found, please check the configuration file.")
        
        # Whether to enable property
        self.enable_properties = self.property_config.get("enabled", False)

        # If property is enabled, load corresponding configuration
        if self.enable_properties:
            raw_properties = self.property_config.get("properties", [])
            self.properties = []  # Optimization problem diversity indicator list, used for extractor operator
            self.property_dict = {}  # PropertyRecord in individual data
            self.property_names = []  # Optimization problem diversity indicator name list, used for extractor operator
            for prop in raw_properties:
                # Check property name
                name = prop.get("name")
                if not name:
                    raise ValueError(f"Property name is not found, please check the property configuration.")

                self.property_names.append(name)
                
                # Convert range to tuple
                prop_cp = dict(prop)
                if isinstance(prop_cp.get("range"), list):
                    prop_cp["range"] = tuple(prop_cp["range"])

                self.properties.append(prop_cp)

                self.property_dict[name] = PropertyRecord(
                    score=0,
                    range=prop_cp.get("range"),
                    for_llm=prop_cp.get("for_llm"),
                    evaluator=prop_cp.get("evaluator"),
                )
        else:
            self.properties = []
            self.property_dict = {}
            self.property_names = []

        # Prompt related
        if config['prompt'].get("role_description", ""):
            self.role_description = config['prompt'].get('role_description', "").rstrip("\n")
        else:
            # Default value
            print(f"[Warning] Role description is not found, using default value.")
            self.role_description = ""

        if config['prompt'].get('problem_description', ""):
            self.problem_description = config['prompt'].get('problem_description', "").rstrip("\n")
        else:
            print(f"[Warning] Problem description is not found, using default value.")
            self.problem_description = "None"

        self.objective_prompt = self._get_objective_prompt(self.objective_list).rstrip("\n")

        if config['prompt'].get('constraints', ""):
            self.constraints = config['prompt'].get('constraints', '').rstrip('\n')
        else:
            print(f"[Warning] Constraints are not found, using default value.")
            self.constraints = "None"

        if config['prompt'].get('expert_insights', ""):
            self.expert_insights = config['prompt'].get("expert_insights", "").rstrip("\n")
        else:
            print(f"[Warning] Expert insights are not found, using default value.")
            self.expert_insights = "None"

        # Prompt template
        self.template = dict()
        self.template_dir = Path(__file__).parent / 'prompt_library'
        self.prompts = ""

    
    def _init_prompt(self):
        """
        Initialize prompt, avoid repeating the same content when iterating
        """
        replacements = {}

        # Role description
        replacements["ROLE_DESCRIPTION"] = self.role_description
        # Problem description
        replacements["PROBLEM_DESCRIPTION"] = self.problem_description
        # Objectives
        replacements["OBJECTIVE"] = self.objective_prompt
        # Constraints
        replacements["CONSTRAINTS"] = self.constraints
        # Expert insights
        replacements["EXPERT_INSIGHTS"] = self.expert_insights
        # Programming language
        replacements["LANGUAGE"] = self.coding_language

        self.prompts = replace_placeholders(self.prompts, replacements=replacements)


    def _get_objective_prompt(self, objectives: List[Dict]) -> str:
        """
        Get optimization objective prompt
        Args:
            objectives (List[Dict]): Optimization objective list
        Returns:
            str: Optimization objective prompt
        """
        if not objectives:
            raise ValueError("objectives is empty or None, please provide at least one objective.")

        obj_prompt = []
        for obj in objectives:
            prompt = []
            obj_name = obj.get("name")  # Optimization objective name
            description = obj.get("description")  # Optimization objective description
            direction = obj.get("direction")  # Optimization direction
            benchmark = obj.get("benchmark")  # Benchmark value

            meta_parts = []
            if direction:
                meta_parts.append("direction: " + str(direction))
            if benchmark:
                meta_parts.append(f"benchmark: {benchmark}")

            if obj_name:
                prompt.append(f"- {obj_name}:")
            else:
                prompt.append("-")

            if description:
                prompt.append(description)

            if meta_parts:
                prompt.append(f"({', '.join(meta_parts)})")

            obj_prompt.append(" ".join(prompt))

        return "\n".join(obj_prompt)

    
    def _get_fitness_prompt(self, fitness: List) -> str:
        if len(fitness) != len(self.objective_list):
            raise ValueError(f"Input fitness list length mismatch the number of optimization objectives: {len(fitness)} != {len(self.objective_list)}")
        
        lines = []
        for idx, (fit_value, obj) in enumerate(zip(fitness, self.objective_list)):
            name = obj.get("name") or f"Objective {idx}"
            benchmark = obj.get("benchmark")

            # Use existing float formatting logic, ensure consistent numerical display
            fit_value = self._format_float_recursive(fit_value)
            benchmark = self._format_float_recursive(benchmark) if benchmark is not None else None

            if benchmark is not None:
                lines.append(f"- {name}: {fit_value}, benchmark: {benchmark}")
            else:
                lines.append(f"- {name}: {fit_value}")

        return "\n".join(lines)

    def _format_float_recursive(self, obj):
        """
        Recursively process data structure, keep 6 significant figures (significant figures)
        """
        if isinstance(obj, float):
            # Note:
            # - round(x, 6) is to keep 6 decimal places, when the value is very small, the effective information will be lost
            # - Here we change to keep 6 significant figures; keep 0/inf/nan as is
            import math

            if obj == 0.0:
                return 0.0
            if math.isinf(obj) or math.isnan(obj):
                return obj

            # Use general format to keep significant figures, then convert back to float
            # 例如：1.23456789e-10 -> 1.23457e-10
            return float(f"{obj:.6g}")
        elif isinstance(obj, dict):
            return {k: self._format_float_recursive(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._format_float_recursive(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._format_float_recursive(item) for item in obj)
        else:
            return obj

    def _dict2prompt(self, data: Dict) -> Optional[str]:
        """
        Convert dictionary to prompt
        
        Args:
            data: Dictionary to be converted
        Returns:
            prompt: Converted string
        """
        if not data:
            print(f"[Warning] _dict2prompt: Input data is empty, None returned.")
            return None

        if not isinstance(data, dict):
            raise TypeError(f"Expected 'dict' type, but got '{type(data).__name__}' instead")

        # Format float
        formatted_data = self._format_float_recursive(data)
        prompt = json.dumps(formatted_data, ensure_ascii=False, indent=2)

        return prompt


    def _list2prompt(self, data: List) -> Optional[str]:
        """
        Convert list to prompt, output list elements one by one, and add "- " prefix
        
        Args:
            data: List to be converted
        Returns:
            prompt: Converted string
        """
        if not data:
            print(f"[Warning] _list2prompt: Input data is empty, None returned.")
            return None

        if not isinstance(data, list):
            raise TypeError(f"Expected 'list' type, but got '{type(data).__name__}' instead")

        # Format float
        formatted_data = self._format_float_recursive(data)
        lines = [f"- {item}" for item in formatted_data]
        prompt = "\n".join(lines)

        return prompt


    def template_loader(self, template_name):
        """
        Load template from prompt_template directory and fill provided keyword parameters
        Args:
            template_name (str): Template file name (without extension)
        Returns:
            filled_template (str): Template string filled with placeholders
        """
        template_path = Path(self.template_dir) / f"{template_name}"

        with open(template_path, 'r', encoding='utf-8') as file:
            template_str = file.read()

        return template_str

    def extract_code_from_llm(self, response: Optional[str], language: str = "python") -> Optional[str]:
        """
        Extract code block content from LLM response

        Args:
            response: LLM response
            language: Code language
        Returns:
            code: Extracted code block
        """
        if not response:
            return None

        response = response.strip()
        language = language.lower().strip()

        # Match template
        pattern = r"```[ \t]*([a-zA-Z0-9_]*)[ \t]*\n?(.*?)(?:```|\Z)"

        matches = re.findall(pattern, response, flags=re.DOTALL | re.IGNORECASE)

        if not matches:
            return response

        # Find the first matching language
        for lang, code in matches:
            if lang.lower().startswith(language):
                return code.strip()

        # Otherwise return the first found code block
        return matches[0][1].strip()

    def _convert_numeric_strings(self, obj):
        """
        Recursively traverse JSON structure, convert strings that can be parsed to int/float, other types remain unchanged
        """
        if isinstance(obj, dict):
            return {k: self._convert_numeric_strings(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._convert_numeric_strings(item) for item in obj]
        if isinstance(obj, str):
            try:
                if "." in obj or "e" in obj.lower():
                    return float(obj)
                return int(obj)
            except ValueError:
                return obj
        return obj


    def extract_json_from_llm(self, response: Optional[str]) -> Optional[Dict]:
        """
        Extract JSON data from LLM response

        Args:
            response: LLM response
        Returns:
            json_data: Extracted JSON data
        """
        if not response:
            return None

        response = response.strip()

        # Match ```json ... ``` code block first
        fence_pattern = r"```[ \t]*json[ \t]*\n?(.*?)(?:```|\Z)"
        match = re.search(fence_pattern, response, flags=re.DOTALL | re.IGNORECASE)

        if match:
            json_str = match.group(1).strip()
        else:
            # Otherwise match the first {} block
            brace_pattern = r"\{.*?\}"
            match = re.search(brace_pattern, response, flags=re.DOTALL)

            if not match:
                return None
            
            json_str = match.group(0).strip()

        try:
            result = json.loads(json_str)
            # Recursively convert numeric types
            return self._convert_numeric_strings(result)
        except json.JSONDecodeError as e:
            print(f"[Warning] The content returned by LLM is not a valid JSON format: {e}")
            return None