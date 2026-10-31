import random
import time


class MigrationManager:
    """
    异步迁移管理器：
    - 独立运行，不阻塞 worker
    - 按 global_counter 周期触发
    - 仅执行读取 elite, copy, insert
    """

    def __init__(
        self,
        config,
        db_url: str,
        island_num: int,
        migration_interval: int,
        migration_top_k: int,
        poll_seconds: float = 0.2,
    ):
        self.config = config
        self.island_num = island_num
        self.migration_interval = migration_interval  # 迁移间隔
        self.migration_top_k = migration_top_k  # 每个子种群单次迁移个体数
        self.poll_seconds = poll_seconds

        # 导入 Population 类
        from population import Population
        self.pop = Population(config, db_url=db_url)


    def _copy_for_migration(self, source_ind, target_island_id: int, iteration_idx: int):
        """
        复制个体用于迁移

        Args:
            source_ind: 源个体
            target_island_id: 目标岛屿 id
            iteration_idx: 当前迭代次数
        """
        from copy import deepcopy
        from population import Individual

        return Individual(
            cell_index=source_ind.cell_index,
            island_id=target_island_id,
            is_elite=False,
            locked=False,
            is_removed=False,
            parent_ids=[source_ind.id] if source_ind.id is not None else [],
            generation=source_ind.generation,
            origin="migration",
            iteration_found=iteration_idx,
            code=source_ind.code,
            fitness=list(source_ind.fitness),
            property_dict=deepcopy(source_ind.property_dict),
            extraction_info=deepcopy(source_ind.extraction_info),
            eval_info=deepcopy(source_ind.eval_info),
            error_info=deepcopy(source_ind.error_info),
        )


    def _select_migrants(self, island_id: int):
        """
        从指定岛屿中随机抽取指定数量的精英个体

        Args:
            island_id: 岛屿 id
        Returns:
            List[Individual]: 抽取的精英个体列表
        """
        where = self.pop._build_where_conditions(
            island_id=island_id,
            is_elite=True,
            is_removed=False,
        )
        elite_inds = self.pop.db.find(where=where)
        if not elite_inds:
            return []

        random.shuffle(elite_inds)
        return elite_inds[:self.migration_top_k]


    def migrate_once(self, iteration_idx: int):
        """
        执行一次岛屿间的个体迁移

        Args:
            iteration_idx: 当前迭代次数
        """
        from population.individual import Individual
        from sqlalchemy import and_

        migrants_by_island = {}
        for src_island in range(self.island_num):
            migrants_by_island[src_island] = self._select_migrants(src_island)

        migrated_count = 0
        skipped_dup = 0
        for src_island in range(self.island_num):
            dst_island = (src_island + 1) % self.island_num
            for src_ind in migrants_by_island[src_island]:
                # 简单去重: 目标岛已有相同 code（且未淘汰）则跳过
                dedup_where = and_(
                    self.pop._build_where_conditions(island_id=dst_island, is_removed=False),
                    Individual.code == src_ind.code,
                )
                if self.pop.db.exists(where=dedup_where):
                    skipped_dup += 1
                    continue

                copied = self._copy_for_migration(src_ind, dst_island, iteration_idx)
                if self.pop.add_individual(copied, dst_island):
                    migrated_count += 1

        print(f"[Migration] Iteration: {iteration_idx}, Migrated: {migrated_count}, Skipped Duplicates: {skipped_dup}")


    def run(self, global_counter, total_iterations: int, stop_event=None):
        """
        个体迁移执行主流程

        Args:
            global_counter: 全局计数器
            total_iterations: 总迭代次数
            stop_event: 可选, 与进化 worker 共享的停止事件 (早停时 set)
        """
        if self.island_num <= 1 or self.migration_interval <= 0 or self.migration_top_k <= 0:
            print("[Warning] 无效的 island_num/migration_interval/migration_top_k, 岛屿个体迁移已禁用")
            return

        print(
            f"[Migration] 岛屿个体迁移管理器已启动, 迁移间隔: {self.migration_interval}, "
            f"单次迁移个体数: {self.migration_top_k}, 岛屿数量: {self.island_num}"
        )
        last_trigger_iter = 0

        while True:
            if stop_event is not None and stop_event.is_set():
                break
            current_iter = int(global_counter.value)  # 当前迭代次数

            while current_iter - last_trigger_iter >= self.migration_interval:
                trigger_iter = last_trigger_iter + self.migration_interval
                try:
                    self.migrate_once(trigger_iter)
                except Exception as e:
                    print(f"[Warning] 个体迁移失败, 迭代次数: {trigger_iter}, 错误: {e}")
                last_trigger_iter = trigger_iter

            if current_iter >= total_iterations:
                break
            time.sleep(self.poll_seconds)
