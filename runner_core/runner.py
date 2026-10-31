import os
from multiprocessing import Process, Value, Lock, Event

from runner_core.island_engine import IslandEngine
from runner_core.migration_manager import MigrationManager


def _parse_env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


# =========================
# Migration Worker（进程入口）
# =========================
def migration_manager_worker(
config,
    db_url,
    island_num: int,
    global_counter,
    total_iterations: int,
    migration_interval: int,
    migration_top_k: int,
    poll_seconds: float = 1,
    stop_evolution=None,
):
    """
    个体迁移进程执行入口

    Args:
        config: 配置
        db_url: 数据库 URL
        island_num: 岛屿数量
        global_counter: 全局计数器
        total_iterations: 总迭代次数
        migration_interval: 迁移间隔
        migration_top_k: 单次迁移个体数
        poll_seconds: 轮询间隔
    """
    manager = MigrationManager(
        config=config,
        db_url=db_url,
        island_num=island_num,
        migration_interval=migration_interval,
        migration_top_k=migration_top_k,
        poll_seconds=poll_seconds,
    )
    manager.run(
        global_counter=global_counter,
        total_iterations=total_iterations,
        stop_event=stop_evolution,
    )

    print("[Migration] 个体迁移执行结束")


# =========================
# Worker（进程入口）
# =========================
def island_worker(
    island_id: int,
    config,
    db_url,
    global_counter,
    counter_lock,
    max_iterations: int,
    batch_size: int = 3,
    stop_evolution=None,
):
    """
    进化流程执行入口

    Args:
        island_id: 岛屿 id
        config: 配置
        db_url: 数据库 URL
        global_counter: 全局计数器
        counter_lock: 计数器锁
        max_iterations: 总迭代次数
        batch_size: 单个 worker 进化执行步长
    """
    print(f"[Island {island_id}] 进化流程执行开始")

    engine = IslandEngine(island_id, config, db_url, stop_event=stop_evolution)
    engine.pop.unlock_island(island_id)

    while True:
        if stop_evolution is not None and stop_evolution.is_set():
            break
        # ===== 批量获取预算（降低锁竞争）=====
        with counter_lock:
            remain = max_iterations - global_counter.value
            current_iteration = global_counter.value
            if remain <= 0:
                break

            local_budget = min(batch_size, remain)
            global_counter.value += local_budget

        # ===== 本地执行 =====
        for i in range(local_budget):
            if stop_evolution is not None and stop_evolution.is_set():
                break
            engine.evolve_once(current_iteration + i)

    print(f"[Island {island_id}] 进化流程执行完毕")


# =========================
# 主入口
# =========================
def run_evolution(
    config,
    db_url,
    total_iterations: int,
    island_num: int,
    batch_size: int = 3,
    stop_evolution=None,
):
    """
    运行进化流程
    
    Args:
        config: 配置
        db_url: 数据库 URL
        total_iterations: 总迭代次数
        island_num: 岛屿数量
        batch_size: 单个 worker 进化执行步长
        stop_evolution: 可选 multiprocessing.Event; 为 None 时内部创建，供早停与 worker 共享
    """
    # Debug mode
    debug_single_process = _parse_env_bool("HYBRIDEVO_DEBUG_RUNNER", default=False)
    if stop_evolution is None:
        stop_evolution = Event()

    if debug_single_process:
        print("\n==================== 启动单进程调试模式 ====================\n")
        if island_num <= 0:
            raise ValueError(f"island_num 必须 > 0, 当前为 {island_num}")

        # 单进程轮询执行各岛屿, 保持 total_iterations 为全局总预算
        engines = [
            IslandEngine(island_id, config, db_url, stop_event=stop_evolution)
            for island_id in range(island_num)
        ]
        for step in range(total_iterations):
            if stop_evolution.is_set():
                break
            island_id = step % island_num
            engines[island_id].evolve_once(step)
        print("\n==================== 单进程调试执行完毕 ====================\n")
        return

    print("\n==================== 启动并行进化 ====================\n")
    print(f"使用模型: {config.get("model")}")

    global_counter = Value("i", 0)
    counter_lock = Lock()
    migration_cfg = config.get("population", {}) or {}
    migration_interval = int(migration_cfg.get("migration_interval", 0))
    migration_top_k = int(migration_cfg.get("migration_top_k", 1))
    migration_poll_seconds = float(migration_cfg.get("migration_poll_seconds", 0.2))

    workers = []
    migration_proc = None

    if island_num > 1 and migration_interval > 0:
        migration_proc = Process(
            target=migration_manager_worker,
            args=(
                config,
                db_url,
                island_num,
                global_counter,
                total_iterations,
                migration_interval,
                migration_top_k,
                migration_poll_seconds,
                stop_evolution,
            ),
        )
        migration_proc.start()

    for island_id in range(island_num):
        p = Process(
            target=island_worker,
            args=(
                island_id,
                config,
                db_url,
                global_counter,
                counter_lock,
                total_iterations,
                batch_size,
                stop_evolution,
            ),
        )
        p.start()
        workers.append(p)

    for p in workers:
        p.join()

    if migration_proc is not None:
        migration_proc.join(timeout=2.0)
        if migration_proc.is_alive():
            migration_proc.terminate()
            migration_proc.join(timeout=1.0)

    if stop_evolution.is_set():
        print("\n========== 进化已提前结束 (命中 ground truth / benchmark 条件) ==========\n")
    else:
        print("\n========== 进化执行完毕 ==========\n")
