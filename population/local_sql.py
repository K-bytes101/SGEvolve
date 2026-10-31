from contextlib import contextmanager
import time
from typing import Iterable, List, Optional, Dict, Any, Sequence, cast, Callable, TypeVar

from sqlmodel import SQLModel, Session, select, create_engine
from sqlalchemy import func, delete, update, insert
from sqlalchemy.exc import OperationalError

from population.individual import Individual

T = TypeVar("T")


class LocalDBManager:
    """
    Simple manager for storing individual information in local SQL database.

    - Default using SQLite database file.
    - Provide basic insert, query/filter, upsert, delete, existence check, count operations.
    """

    def __init__(self, db_url: str = "sqlite:///./local_population.db", echo: bool = False) -> None:
        """
        Initialize database connection and metadata.

        :param db_url: Database connection URL, default using SQLite file in current directory.
        :param echo: Whether to print SQL statements, for debugging.
        """
        self.engine = create_engine(db_url, echo=echo)
        self._write_retry_times = 3
        self._write_retry_base_seconds = 0.05
        self._busy_timeout_ms = 5000
        self._configure_sqlite_pragmas()
        # Create all tables defined by SQLModel
        SQLModel.metadata.create_all(self.engine)
    

    def _configure_sqlite_pragmas(self):
        """Configure SQLite performance/security parameters: WAL improves concurrent write, foreign_keys ensures foreign key constraints."""
        try:
            with self.engine.connect() as conn:
                conn.exec_driver_sql("PRAGMA journal_mode=WAL;")
                conn.exec_driver_sql("PRAGMA synchronous=NORMAL;")
                conn.exec_driver_sql("PRAGMA foreign_keys=ON;")
                conn.exec_driver_sql(f"PRAGMA busy_timeout={self._busy_timeout_ms};")
        except Exception as e:
            print(f"Failed to set PRAGMA: {e}")


    def _is_sqlite_locked_error(self, err: Exception) -> bool:
        if isinstance(err, OperationalError):
            msg = str(err).lower()
            return (
                "database is locked" in msg
                or "database table is locked" in msg
                or "database schema is locked" in msg
            )
        return False


    def _execute_write_with_retry(self, op_name: str, fn: Callable[[], T]) -> T:
        for attempt in range(self._write_retry_times + 1):
            try:
                return fn()
            except Exception as err:
                if (not self._is_sqlite_locked_error(err)) or attempt >= self._write_retry_times:
                    raise
                sleep_seconds = self._write_retry_base_seconds * (2 ** attempt)
                print(f"Warning: {op_name} encountered SQLite lock conflict, will retry in {sleep_seconds:.2f}s")
                time.sleep(sleep_seconds)
        raise RuntimeError(f"{op_name} retry process ended abnormally")


    @contextmanager
    def _session(self) -> Session:
        """
        Provide a context-managed Session, automatically commit / rollback / close
        """
        with Session(self.engine) as sess:
            try:
                yield sess
                sess.commit()
            except Exception as e:
                sess.rollback()
                raise e

    # ---------- Core CRUD operations ----------

    def add(self, individual: Individual) -> Individual:
        """
        Insert a single individual
        """
        def _op() -> Individual:
            with Session(self.engine) as sess:
                sess.add(individual)
                sess.commit()
                sess.refresh(individual)
                # Convert to independent Individual instance before Session is closed, avoid DetachedInstanceError
                return Individual.model_validate(individual.model_dump())
        return self._execute_write_with_retry("add", _op)


    def bulk_add(self, individuals: Iterable[Individual]) -> List[Individual]:
        """
        Batch insert multiple individuals.
        """
        individuals = list(individuals)
        if not individuals:
            return []

        def _op() -> List[Individual]:
            with Session(self.engine) as sess:
                sess.add_all(individuals)
                sess.commit()
                # After commit(), all objects need to be refreshed to ensure all attributes are loaded (avoid attribute loss caused by expire)
                for ind in individuals:
                    sess.refresh(ind)
                # Convert to independent Individual instance before Session is closed, avoid DetachedInstanceError
                return [Individual.model_validate(ind.model_dump()) for ind in individuals]
        return self._execute_write_with_retry("bulk_add", _op)


    def get_by_id(self, ind_id: int) -> Optional[Individual]:
        """
        Query individual by primary key id
        """
        with Session(self.engine) as sess:
            db_obj = sess.get(Individual, ind_id)
            if db_obj is None:
                return None
            # Convert to independent Individual instance before Session is closed, avoid DetachedInstanceError
            return Individual.model_validate(db_obj.model_dump())


    def find(self, *, where=None, order_by=None, limit=None, offset=None) -> List[Individual]:
        """
        General query interface, support where conditions, sorting, pagination
        """
        stmt = select(Individual)
        if where is not None:
            stmt = stmt.where(where)
        if order_by is not None:
            stmt = stmt.order_by(order_by)
        if offset is not None:
            stmt = stmt.offset(offset)
        if limit is not None:
            stmt = stmt.limit(limit)

        with Session(self.engine) as sess:
            orm_results = sess.exec(stmt).all()
            # Convert to independent Individual instance before Session is closed, avoid DetachedInstanceError
            return [Individual.model_validate(ind.model_dump()) for ind in orm_results]


    def update_ind_by_dict(self, ind_id, update_data: Dict[str, Any]) -> Optional[Individual]:
        """
        Update an existing individual by primary key id

        Args:
            ind_id: The id of the individual to be updated
            update_data: Update data
        Returns:
            Optional[Individual]: Updated individual
        """
        def _op() -> Optional[Individual]:
            with Session(self.engine) as sess:
                db_obj = sess.get(Individual, ind_id)

                if db_obj is None:
                    print(f"Warning: No individual record found with id={ind_id}")
                    return None

                # Type assertion: After None check, db_obj must be Individual type
                db_obj = cast(Individual, db_obj)

                # Apply update dictionary to object
                for key, value in update_data.items():
                    setattr(db_obj, key, value)

                sess.add(db_obj)
                sess.commit()
                sess.refresh(db_obj)

                return Individual.model_validate(db_obj.model_dump())
        return self._execute_write_with_retry("update_ind_by_dict", _op)


    def update_individual(self, ind: Individual) -> Optional[Individual]:
        """
        Update an existing individual by individual
        """
        if self.get_by_id(ind.id) is None:
            print(f"Warning: No individual record found with id={ind.id} when updating individual")
            return None

        def _op() -> Optional[Individual]:
            with Session(self.engine) as sess:
                stmt = (
                    update(Individual)
                    .where(Individual.id == ind.id)
                    .values(
                        # Do not update attributes managed by other methods, such as locked, island_id, etc.
                        parent_ids=ind.parent_ids,
                        cell_index=ind.cell_index,
                        is_elite=ind.is_elite,
                        generation=ind.generation,
                        origin=ind.origin,
                        iteration_found=ind.iteration_found,
                        code=ind.code,
                        fitness=ind.fitness,
                        property_dict=ind.property_dict,
                        extraction_info=ind.extraction_info,
                        eval_info=ind.eval_info,
                        error_info=ind.error_info,
                    )
                )

                sess.exec(stmt)
                sess.commit()

                updated_ind = sess.get(Individual, ind.id)
                return Individual.model_validate(updated_ind.model_dump())
        return self._execute_write_with_retry("update_individual", _op)

    
    def remove_by_id(self, ind_id: int) -> bool:
        """
        Mark an individual as removed by primary key id
        """
        if self.get_by_id(ind_id) is None:
            print(f"Warning: No individual record found with id={ind_id} when removing individual")
            return False
        
        def _op() -> bool:
            with Session(self.engine) as sess:
                stmt = update(Individual).where(Individual.id == ind_id).values(is_removed=True)  # type: ignore
                res = sess.exec(stmt)
                sess.commit()
                return res.rowcount > 0  # Return whether removal is successful
        return self._execute_write_with_retry("remove_by_id", _op)


    def delete_by_id(self, ind_id: int) -> bool:
        """
        Delete an individual by primary key id, return whether deletion is successful
        """
        def _op() -> bool:
            with Session(self.engine) as sess:
                stmt = delete(Individual).where(Individual.id == ind_id)  # type: ignore
                res = sess.exec(stmt)
                sess.commit()
                return res.rowcount > 0  # Return whether deletion is successful
        return self._execute_write_with_retry("delete_by_id", _op)


    def exists(self, where=None) -> bool:
        """
        Check if there is an individual that satisfies the conditions.
        """
        with Session(self.engine) as sess:
            stmt = (
                select(Individual)
                .where(where)
                .limit(1)
            )
            return sess.exec(stmt).first() is not None


    def count(self, where=None) -> int:
        """
        Count the number of individuals that satisfy the conditions
        """
        with Session(self.engine) as sess:
            stmt = (
                select(func.count()).select_from(select(Individual).where(where).subquery())
                if where is not None
                else select(func.count(Individual.id))
            )
            return int(sess.exec(stmt).one())

    
    def sample_and_lock(self, num: int = 1, where: Any = None) -> Optional[List[Individual]]:
        """
        Sample and lock an individual according to the conditions
        """
        def _op() -> Optional[List[Individual]]:
            with Session(self.engine) as sess:
                # Construct sub-query to get the list of individual ids to be sampled
                subq = select(Individual.id).where(Individual.locked == False)
                if where is not None:
                    subq = subq.where(where)
                subq = subq.order_by(func.random()).limit(num)

                # Construct atomic update statement to lock the individuals to be sampled
                stmt = (
                    update(Individual)
                    .where(Individual.id.in_(subq))
                    .values(locked=True)
                    .returning(Individual)
                )

                # Execute update operation
                orm_rows = sess.exec(stmt).scalars().all()

                if not orm_rows:
                    # No individuals sampled, return None
                    return None

                detached = [Individual.model_validate(ind.model_dump()) for ind in orm_rows]
                sess.commit()
                return detached
        return self._execute_write_with_retry("sample_and_lock", _op)


    def unlock(self, ind_id: int) -> bool:
        """
        Unlock an individual by primary key id, return whether unlocking is successful
        """
        def _op() -> bool:
            with Session(self.engine) as sess:
                stmt = update(Individual).where(Individual.id == ind_id).values(locked=False)  # type: ignore
                res = sess.exec(stmt)
                sess.commit()
                return res.rowcount > 0  # Return whether unlocking is successful
        return self._execute_write_with_retry("unlock", _op)


    def clear_all(self) -> bool:
        """
        Clear all individuals
        """
        def _op() -> bool:
            with Session(self.engine) as sess:
                stmt = delete(Individual)  # type: ignore
                res = sess.exec(stmt)
                sess.commit()
                return res.rowcount > 0  # Return whether deletion is successful
        return self._execute_write_with_retry("clear_all", _op)
