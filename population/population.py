from typing import List, Tuple, Any, Optional
from enum import Enum
import numpy as np
from sqlalchemy import and_, true

from population.individual import PropertyRecord, Individual
from population.local_sql import LocalDBManager
from utils import format_obj_list


class OptType(str, Enum):
    """
    Optimization type, single-objective optimization or multi-objective optimization
    """
    SINGLE = "single_objective"
    MULTI = "multi_objective"


class Population:
    """
    Population management class
    Implement MAP-Elites and Island Population mechanism
    """

    def __init__(self, config, db_url: str = "sqlite:///./local_population.db"):
        # Configs
        self.config = config  # Problem configuration
        self.opt_type = OptType.SINGLE  # Optimization type, single-objective optimization or multi-objective optimization

        self.pop_config = config['population'] or {}
        if not self.pop_config:
            raise ValueError("population configuration not found")

        # Optimization objective list
        self.objective_list = format_obj_list(self.config['prompt'].get('objectives', []))
        if not self.objective_list:
            raise ValueError("Objectives are not found, please check the objectives configuration.")
        self.objective_direction = np.array(
            [1 if obj["direction"] == "min" else -1 for obj in self.objective_list],
            dtype=np.int8
        )
        # Optimization objective number
        self.objective_num = len(self.objective_list)

        # Determine optimization type
        if self.objective_num == 1:
            self.opt_type = OptType.SINGLE
        elif self.objective_num > 1:
            self.opt_type = OptType.MULTI
        else:
            raise ValueError("Objective number is not valid")

        # Population parameters
        self.iterations = self.pop_config.get("iterations")  # Iteration number
        self.pop_size = self.pop_config.get("population_size")  # Population size of each island
        self.island_num = self.pop_config.get("island_num")  # Island number
        self.bins_num = self.pop_config.get("bins_num")  # Number of bins per dimension in MAP-Elites
        self.ind_num_per_cell = self.pop_config.get("ind_num_per_cell")  # Number of individuals per cell
        self.pareto_archive_size = self.pop_config.get("pareto_archive_size", 10)  # Pareto archive size

        # Database
        self.db = LocalDBManager(db_url)


    # ---------- Database methods ----------
    def _build_where_conditions(
        self, 
        *, 
        cell_index: Optional[str] = None,
        island_id: Optional[int] = None, 
        is_elite: Optional[bool] = None, 
        locked: Optional[bool] = None,
        is_removed: Optional[bool] = None,
    ) -> Any:
        """
        Build where query expression
        Args:
            cell_index: cell index
            island_id: island id
            is_elite: whether is elite
            locked: whether is locked
        Returns:
            where query expression
        """
        conditions = []

        if cell_index is not None:
            conditions.append(Individual.cell_index == cell_index)
        if island_id is not None:
            conditions.append(Individual.island_id == island_id)
        if is_elite is not None:
            conditions.append(Individual.is_elite == is_elite)
        if locked is not None:
            conditions.append(Individual.locked == locked)
        if is_removed is not None:
            conditions.append(Individual.is_removed == is_removed)

        where = and_(*conditions) if conditions else true()
        return where


    def get_db_manager(self) -> LocalDBManager:
        """
        Get database manager
        """
        return self.db

    
    # ---------- Built-in helper methods ----------
    def _get_cell_index(self, ind: Individual) -> str:
        """
        Calculate the position of the individual in the bin according to the attribute values and ranges in the property_dict
        Args:
            ind: individual object
        Returns:
            str: A string of the form "bin1, bin2, ..." representing the bin position of each attribute dimension
        """
        if not ind.property_dict:
            # If there are no attributes, return empty string or default value
            return ""
        
        bin_positions = []
        
        # Traverse the attributes in the default order (insertion order)
        for _, prop_record in ind.property_dict.items():
            score = prop_record.score
            prop_range = prop_record.range
            
            # Check if the range is valid
            if len(prop_range) != 2:
                # If the range is invalid, use default bin 1
                bin_positions.append(1)
                continue
            
            min_val, max_val = prop_range
            
            # Handle boundary cases: if min == max, all values are in the same bin
            if min_val == max_val:
                bin_pos = 1
            else:
                # Split the range into bins_num intervals, map the score to the corresponding bin position
                bin_width = (max_val - min_val) / self.bins_num
                
                # Calculate the bin where the score is located (starting from 1)
                if score < min_val:
                    bin_pos = 1  # Less than the minimum value, put in the first bin
                elif score >= max_val:
                    bin_pos = self.bins_num  # Greater than or equal to the maximum value, put in the last bin
                else:
                    # Calculate the bin position: floor((score - min_val) / bin_width) + 1
                    bin_pos = int((score - min_val) / bin_width) + 1
                    # Ensure bin_pos is within the valid range [1, bins_num]
                    bin_pos = min(bin_pos, self.bins_num)
            
            bin_positions.append(bin_pos)
        
        # Return a string of the form "bin1, bin2, ..."
        return ", ".join(str(bin_pos) for bin_pos in bin_positions)

    
    def _init_elite(self, ind_list: List[Individual], island_id: int) -> bool:
        """
        Initialize elite individuals according to ind_list

        Args:
            ind_list: individual list
            island_id: island id
        """
        if self.opt_type == OptType.SINGLE:  # Single-objective optimization
            # Store how many cell_index
            cell_index_set = set()
            for ind in ind_list:
                cell_index_set.add(ind.cell_index)

            # For each cell_index, keep the top ind_num_per_cell individuals with the highest fitness
            for cell_index in cell_index_set:
                where = self._build_where_conditions(cell_index=cell_index, island_id=island_id, is_removed=False)
                inds_with_same_cell_index = self.db.find(where=where)
                if not inds_with_same_cell_index:
                    continue

                # Sort
                reverse = True if self.objective_direction[0] == -1 else False
                sorted_inds = sorted(
                    inds_with_same_cell_index,
                    key=lambda x: float(x.fitness[0]),
                    reverse=reverse
                )
                elite_inds = sorted_inds[:self.ind_num_per_cell] if len(sorted_inds) > self.ind_num_per_cell else sorted_inds
                
                # Batch update these individuals to be elite
                for elite_ind in elite_inds:
                    if elite_ind.error_info:
                        # There is an error
                        continue
                    else:
                        if not self.db.update_ind_by_dict(elite_ind.id, {"is_elite": True}):
                            print(f"Warning: Failed to initialize elite individual {elite_ind.id}")
                            return False
        elif self.opt_type == OptType.MULTI:  # Multi-objective optimization
            # Build a Pareto archive
            valid_inds: List[Individual] = []
            fitness_rows: List[np.ndarray] = []

            # Filter valid individuals
            for ind in ind_list:
                if len(ind.fitness) != self.objective_num:
                    continue
                fitness_row = np.asarray(ind.fitness, dtype=np.float64)
                valid_inds.append(ind)
                fitness_rows.append(fitness_row)

            if not valid_inds:
                return True

            # Build (N, M) fitness matrix, and unify it to the minimization problem, N: number of individuals, M: number of objectives
            fitness_nm = np.vstack(fitness_rows) * self.objective_direction.reshape(1, -1)

            # Dominance relation matrix dominates[j, i] represents j dominates i
            all_le = np.all(fitness_nm[:, None, :] <= fitness_nm[None, :, :], axis=2)
            any_lt = np.any(fitness_nm[:, None, :] < fitness_nm[None, :, :], axis=2)
            dominates = all_le & any_lt

            dominated_mask = np.any(dominates, axis=0)
            pareto_indices = np.where(~dominated_mask)[0]

            # When the Pareto archive exceeds the limit, based on the crowding distance to remove the most crowded individuals
            if self.pareto_archive_size and self.pareto_archive_size > 0 and len(pareto_indices) > self.pareto_archive_size:
                selected_indices = pareto_indices.copy()
                archive_size = int(self.pareto_archive_size)

                while len(selected_indices) > archive_size:
                    front_fitness = fitness_nm[selected_indices]  # (K, M)
                    k, m = front_fitness.shape
                    crowding_distance = np.zeros(k, dtype=np.float64)

                    for obj_idx in range(m):
                        sorted_local = np.argsort(front_fitness[:, obj_idx], kind="mergesort")
                        crowding_distance[sorted_local[0]] = np.inf
                        crowding_distance[sorted_local[-1]] = np.inf

                        obj_values = front_fitness[sorted_local, obj_idx]
                        obj_span = obj_values[-1] - obj_values[0]
                        if obj_span <= 0:
                            continue

                        internal_local = sorted_local[1:-1]
                        if len(internal_local) > 0:
                            prev_vals = obj_values[:-2]
                            next_vals = obj_values[2:]
                            crowding_distance[internal_local] += (next_vals - prev_vals) / obj_span

                    # Remove the individual with the smallest crowding distance; if tied, stable select the first one
                    remove_local_idx = int(np.argmin(crowding_distance))
                    selected_indices = np.delete(selected_indices, remove_local_idx)

                pareto_indices = selected_indices

            elite_id_set = {valid_inds[idx].id for idx in pareto_indices}

            # Unify the is_elite mark for the individuals on the island
            for ind in ind_list:
                target_is_elite = ind.id in elite_id_set
                if ind.is_elite != target_is_elite:
                    if not self.db.update_ind_by_dict(ind.id, {"is_elite": target_is_elite}):
                        print(f"Warning: Failed to initialize elite individual {ind.id}")
                        return False

        return True


    def _is_elite(self, ind: Individual) -> bool:
        """
        Determine if the individual is elite
        """
        is_elite, _ = self._plan_elite_update(ind)
        return is_elite


    def _plan_elite_update(self, ind: Individual) -> Tuple[bool, set[int]]:
        """
        Calculate the elite change plan

        Returns:
            Tuple[bool, set[int]]:
            - bool: Whether the new individual should be elite
            - set[int]: The set of old individual ids that need to be canceled from the elite state
        """
        # Convert fitness to the minimization problem
        fitness = np.array(ind.fitness, dtype=np.float64) * self.objective_direction
        cell_index = ind.cell_index
        island_id = ind.island_id
        error_info = ind.error_info

        if error_info:
            # There is error information, return False
            return False, set()

        if self.opt_type == OptType.SINGLE:
            where = self._build_where_conditions(cell_index=cell_index, island_id=island_id, is_elite=True, is_removed=False)
            # Check if the grid is full
            ind_elites = self.db.find(where=where)

            if not ind_elites or len(ind_elites) < self.ind_num_per_cell:
                # The number of elite individuals with this cell_index is insufficient, return True
                return True, set()
            else:
                # Unify to the minimization problem
                elite_fitness_raw = np.array([float(elite.fitness[0]) for elite in ind_elites], dtype=np.float64)
                elite_fitness_min = elite_fitness_raw * self.objective_direction

                # Find the worst elite
                worst_idx = int(np.argmax(elite_fitness_min))
                worst_elite = ind_elites[worst_idx]
                worst_fitness = float(elite_fitness_min[worst_idx])

                if fitness[0] <= worst_fitness:
                    demote_ids = {worst_elite.id} if worst_elite.id is not None else set()
                    return True, demote_ids
                return False, set()
            
        elif self.opt_type == OptType.MULTI:
            # Query elite individuals
            where = self._build_where_conditions(island_id=island_id, is_elite=True, is_removed=False)
            elite_inds = self.db.find(where=where)

            if not elite_inds:
                return True, set()

            # Check if the fitness value is valid
            if len(fitness) != self.objective_num:
                raise ValueError("Error: Fitness length is not valid")

            # Build the fitness value matrix, and unify it to the minimization problem
            elite_fitness_nm = np.array(
                [np.asarray(elite.fitness, dtype=np.float64) for elite in elite_inds],
                dtype=np.float64
            ) * self.objective_direction.reshape(1, -1)

            # If any existing elite dominates the new individual, then the new individual is not elite
            elite_dominate_ind = np.all(elite_fitness_nm <= fitness.reshape(1, -1), axis=1) & \
                np.any(elite_fitness_nm < fitness.reshape(1, -1), axis=1)
            if np.any(elite_dominate_ind):
                return False, set()

            # The old elite dominated by the new individual needs to be removed from the archive
            ind_dominate_elite = np.all(fitness.reshape(1, -1) <= elite_fitness_nm, axis=1) & \
                np.any(fitness.reshape(1, -1) < elite_fitness_nm, axis=1)
            remove_elite_ids = {elite_inds[idx].id for idx in np.where(ind_dominate_elite)[0]}

            # The retained elite individuals
            keep_mask = ~ind_dominate_elite
            kept_elites = [elite for i, elite in enumerate(elite_inds) if keep_mask[i]]
            kept_fitness_nm = elite_fitness_nm[keep_mask]

            archive_size = self.pareto_archive_size
            if 0 < archive_size < (len(kept_elites) + 1):
                # Use the crowding distance to decide the淘汰对象 when the archive exceeds the limit (boundary points are inf)
                combined_ids = [elite.id for elite in kept_elites] + [None]  # None 表示新个体
                combined_fitness = np.vstack([kept_fitness_nm, fitness.reshape(1, -1)])

                while len(combined_ids) > archive_size:
                    k, m = combined_fitness.shape
                    crowding_distance = np.zeros(k, dtype=np.float64)

                    for obj_idx in range(m):
                        sorted_local = np.argsort(combined_fitness[:, obj_idx], kind="mergesort")
                        crowding_distance[sorted_local[0]] = np.inf
                        crowding_distance[sorted_local[-1]] = np.inf

                        obj_values = combined_fitness[sorted_local, obj_idx]
                        obj_span = obj_values[-1] - obj_values[0]
                        if obj_span <= 0:
                            continue

                        internal_local = sorted_local[1:-1]
                        if len(internal_local) > 0:
                            prev_vals = obj_values[:-2]
                            next_vals = obj_values[2:]
                            crowding_distance[internal_local] += (next_vals - prev_vals) / obj_span

                    remove_idx = int(np.argmin(crowding_distance))
                    remove_id = combined_ids[remove_idx]
                    if remove_id is None:
                        # The new individual is eliminated in the crowding distance clipping, do not execute the old elite demotion
                        return False, set()

                    remove_elite_ids.add(remove_id)
                    combined_fitness = np.delete(combined_fitness, remove_idx, axis=0)
                    combined_ids.pop(remove_idx)

            return True, remove_elite_ids

        return False, set()


    def _apply_elite_demotions(self, demote_ids: set[int]) -> bool:
        """
        Execute the old elite demotion.
        """
        for old_id in demote_ids:
            if not self.db.update_ind_by_dict(old_id, {"is_elite": False}):
                print(f"Warning: Failed to update elite individual {old_id}")
                return False
        return True


    def _maintain_cell_capacity(self, island_id: int, cell_index: Optional[str] = None) -> bool:
        """
        Maintain the cell capacity

        Args:
            island_id: island id
            cell_index: cell index, if cell_index is None, maintain all cell capacities
        Returns:
            bool: Whether the cell capacity is successfully maintained
        """
        # TODO
        pass


    # ---------- 种群管理方法 ----------
    def init_population(self, ind_list: List[Individual], island_id: int) -> bool:
        """
        Add the initial individual list to the database, need to complete the initialization of cell_index and Elite

        Args:
            ind_list: initial individual list
            island_id: island id
        Returns:
            bool: Whether the initialization is successful
        """
        # Initialize cell_index, island_id
        island_id = island_id % self.island_num  # Ensure island_id is within the valid range

        for ind in ind_list:
            ind.island_id = island_id
            if self.opt_type == OptType.SINGLE:
                # Single-objective, use cell_index to maintain MAP-Elites
                ind.cell_index = self._get_cell_index(ind)
            elif self.opt_type == OptType.MULTI:
                # Multi-objective, directly use the Pareto archive to maintain diversity
                ind.cell_index = ""

        # Store in the database
        if not self.db.bulk_add(ind_list):
            return False

        # Initialize elite
        if not self._init_elite(ind_list, island_id):
            return False

        return True

    
    def sample_from_island(self, num: int, island_id: int, is_elite: bool = False) -> Optional[List[Individual]]:
        """
        Sample a specified number of individuals from a specified island
        Args:
            num: number of individuals to sample
            island_id: island id
            is_elite: whether the individual is elite
        Returns:
            Optional[List[Individual]]: The sampled individual list
        """
        if num <= 0:
            raise ValueError("The number of individuals to sample must be greater than 0")

        where = self._build_where_conditions(island_id=island_id, is_elite=is_elite, is_removed=False)
        inds = self.db.sample_and_lock(num=num, where=where)

        if inds is None:
            print(f"Warning: No individuals sampled from island {island_id}")
            return None

        if len(inds) != num:
            print(f"Warning: No individuals sampled from island {island_id}")
            return inds
        
        return inds


    def mutation_sampling(self, island_id: int) -> Optional[List[Individual]]:
        """
        Sample a single individual from a specified island for mutation

        Args:
            island_id: island id
        Returns:
            Optional[List[Individual]]: The sampled individual list, len = 1
        """
        where = self._build_where_conditions(island_id=island_id, is_removed=False)
        ind = self.db.sample_and_lock(num=1, where=where)

        if not ind:
            print(f"Warning: No individuals sampled from island {island_id} for mutation")
            return None
        
        return ind


    def crossover_sampling(self, island_id: int) -> Optional[List[Individual]]:
        """
        Sample two individuals from a specified island for crossover

        Args:
            island_id: island id
        Returns:
            Optional[List[Individual]]: The sampled individual list
        """
        where = self._build_where_conditions(island_id=island_id, is_elite=True, is_removed=False)
        inds = self.db.sample_and_lock(num=2, where=where)

        if not inds or len(inds) < 2:
            # When sampling is insufficient, release the locked individuals to avoid lock leaks
            if inds:
                for ind in inds:
                    self.db.unlock(ind.id)
            print(f"Warning: No individuals sampled from island {island_id} for crossover")
            return None
        
        return inds

    
    def balance_sampling(self, island_id: int) -> Optional[List[Individual]]:
        """
        Sample one elite individual and one non-elite individual from a specified island

        Args:
            island_id: island id
        Returns:
            Optional[List[Individual]]: The sampled individual list
        """
        inds = []
        where = self._build_where_conditions(island_id=island_id, is_elite=True, is_removed=False)
        elite_ind = self.db.sample_and_lock(num=1, where=where)
        if not elite_ind:
            print(f"Warning: No elite individuals sampled from island {island_id}")
            return None
        else:
            inds.append(elite_ind[0])

        where = self._build_where_conditions(island_id=island_id, is_elite=False, is_removed=False)
        non_elite_ind = self.db.sample_and_lock(num=1, where=where)
        if not non_elite_ind:
            # When the second sampling fails, release the first locked elite individual to avoid lock leaks
            self.db.unlock(elite_ind[0].id)
            print(f"Warning: No non-elite individuals sampled from island {island_id}")
            return None
        else:
            inds.append(non_elite_ind[0])

        return inds


    def add_individual(self, ind: Individual, island_id: int) -> bool:
        """
        Add an individual to the database, maintain MAP-Elites / Pareto Archive

        Args:
            ind: individual to add
            island_id: island id
        Returns:
            bool: Whether the individual is successfully added
        """
        ind_fitness = np.array(ind.fitness, dtype=np.float64) * self.objective_direction

        # Query the island population size
        where = self._build_where_conditions(island_id=island_id, is_removed=False)
        island_pop_size = self.db.count(where=where)

        if self.opt_type == OptType.SINGLE:  # Single-objective, maintain the cell capacity is relatively simple
            # 确定 cell_index is_elite属性
            ind.cell_index = self._get_cell_index(ind)  # Get cell_index
            ind.is_elite, demote_ids = self._plan_elite_update(ind)  # Determine if the individual is elite

            # Add individual
            if island_pop_size < self.pop_size:
                # If the island population size is less than pop_size, add directly
                if not self.db.add(ind):
                    print(f"Warning: Failed to add individual {ind.id}")
                    return False
                if ind.is_elite and not self._apply_elite_demotions(demote_ids):
                    print(f"Warning: Failed to downgrade elite individual {demote_ids} when adding individual {ind.id}")
                    return False
            else:
                # Otherwise, determine whether to add based on fitness and is_elite
                where = self._build_where_conditions(island_id=island_id, is_elite=False, is_removed=False)
                inds_in_island = self.db.find(where=where)
                
                # Find the worst individual
                ind_fitness_raw = np.array([float(ind.fitness[0]) for ind in inds_in_island], dtype=np.float64)
                ind_fitness_min = ind_fitness_raw * self.objective_direction

                worst_idx = int(np.argmax(ind_fitness_min))
                worst_ind = inds_in_island[worst_idx]
                worst_fitness = float(ind_fitness_min[worst_idx])

                if ind_fitness < worst_fitness:
                    # Remove the worst individual
                    if not self.db.remove_by_id(worst_ind.id):
                        print(f"Warning: Failed to remove individual {worst_ind.id} when adding new individual")
                        return False
                        
                    # 添加新个体
                    if not self.db.add(ind):
                        print(f"Warning: Failed to add individual {ind.id}")
                        return False
                    if ind.is_elite and not self._apply_elite_demotions(demote_ids):
                        return False
                else:
                    # The individual is not good enough, mark is_removed as True, then put it into the population
                    ind.is_removed = True
                    if not self.db.add(ind):
                        print(f"Warning: Failed to add individual {ind.id}, ind.is_removed = True")
                        return False
                    return True
        elif self.opt_type == OptType.MULTI:
            # Multi-objective optimization, maintain the population size
            ind.cell_index = ""
            if island_pop_size < self.pop_size:
                # If the island population size is less than pop_size, add directly
                ind.is_elite, demote_ids = self._plan_elite_update(ind)
                if not self.db.add(ind):
                    print(f"Warning: Failed to add individual {ind.id}")
                    return False
                if ind.is_elite and not self._apply_elite_demotions(demote_ids):
                    return False
            else:
                where = self._build_where_conditions(island_id=island_id, is_removed=False)
                inds_in_island = self.db.find(where=where)

                # Pareto Rank + Crowding Distance to maintain the population
                all_inds = inds_in_island + [ind]
                if len(all_inds) <= 1:
                    return False

                fitness_nm = np.array(
                    [np.asarray(ind.fitness, dtype=np.float64) for ind in all_inds],
                    dtype=np.float64
                )
                if fitness_nm.shape[1] != self.objective_num:
                    return False
                fitness_nm = fitness_nm * self.objective_direction.reshape(1, -1)

                # Non-dominated sorting, take the worst front
                remaining_mask = np.ones(len(all_inds), dtype=bool)
                fronts: List[np.ndarray] = []
                while np.any(remaining_mask):
                    rem_idx = np.where(remaining_mask)[0]
                    rem_fit = fitness_nm[rem_idx]
                    all_le = np.all(rem_fit[:, None, :] <= rem_fit[None, :, :], axis=2)
                    any_lt = np.any(rem_fit[:, None, :] < rem_fit[None, :, :], axis=2)
                    dominates = all_le & any_lt
                    dominated_mask = np.any(dominates, axis=0)
                    current_front = rem_idx[~dominated_mask]
                    fronts.append(current_front)
                    remaining_mask[current_front] = False

                worst_front = fronts[-1]
                if len(worst_front) == 1:
                    remove_idx = int(worst_front[0])
                else:
                    front_fitness = fitness_nm[worst_front]  # (K, M)
                    k, m = front_fitness.shape
                    crowding_distance = np.zeros(k, dtype=np.float64)

                    for obj_idx in range(m):
                        sorted_local = np.argsort(front_fitness[:, obj_idx], kind="mergesort")
                        crowding_distance[sorted_local[0]] = np.inf
                        crowding_distance[sorted_local[-1]] = np.inf

                        obj_values = front_fitness[sorted_local, obj_idx]
                        obj_span = obj_values[-1] - obj_values[0]
                        if obj_span <= 0:
                            continue

                        internal_local = sorted_local[1:-1]
                        if len(internal_local) > 0:
                            prev_vals = obj_values[:-2]
                            next_vals = obj_values[2:]
                            crowding_distance[internal_local] += (next_vals - prev_vals) / obj_span

                    remove_local = int(np.argmin(crowding_distance))
                    remove_idx = int(worst_front[remove_local])

                # The new individual is not dominated, set is_removed to True, then put it into the population
                if remove_idx == len(all_inds) - 1:
                    ind.is_removed = True
                    if not self.db.add(ind):
                        print(f"Warning: Failed to add individual {ind.id}, ind.is_removed = True")
                        return False
                    return True

                # First remove the old individual, then determine if the new individual is elite based on the current elite archive
                remove_ind = all_inds[remove_idx]
                if not self.db.remove_by_id(remove_ind.id):
                    print(f"Warning: Failed to remove individual {remove_ind.id} when adding new individual")
                    return False

                ind.is_elite, demote_ids = self._plan_elite_update(ind)
                if not self.db.add(ind):
                    print(f"Warning: Failed to add individual {ind.id}")
                    return False
                if ind.is_elite and not self._apply_elite_demotions(demote_ids):
                    print(f"Warning: Failed to downgrade elite individual {demote_ids} when adding individual {ind.id}")
                    return False
        return True

    
    def update_individual(self, ind: Individual) -> Optional[Individual]:
        """
        Update an individual
        """
        return self.db.update_individual(ind)

    
    def unlock_sampled_ind(self, ind_id: int) -> bool:
        """
        Unlock the sampled individual
        """
        return self.db.unlock(ind_id)

    
    def unlock_island(self, island_id: int) -> int:
        """
        Unlock all locked individuals in island_id

        Args:
            island_id: island id
        Returns:
            int: The number of unlocked individuals
        """
        island_id = island_id % self.island_num  # Ensure island_id is within the valid range
        where = self._build_where_conditions(island_id=island_id, locked=True)
        locked_inds = self.db.find(where=where)
        count = 0
        for ind in locked_inds:
            if not self.db.unlock(ind.id):
                print(f"Warning: Failed to unlock individual {ind.id}")
            else:
                count += 1
        return count


