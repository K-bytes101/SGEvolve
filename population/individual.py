from typing import Tuple, List, Dict, Optional, Any, Type, TypeVar
from sqlmodel import SQLModel, Field, Column, JSON
from sqlalchemy.types import TypeDecorator, TEXT
from pydantic import TypeAdapter

class PydanticJSON(TypeDecorator):
    """
    Custom type: Implement automatic conversion between Pydantic model and database JSON string
    """
    impl = TEXT
    cache_ok = True

    def __init__(self, pydantic_type: Any):
        super().__init__()
        # TypeAdapter is a powerful tool introduced by Pydantic v2, used to parse complex types
        self.adapter = TypeAdapter(pydantic_type)

    def process_bind_param(self, value, dialect):
        """Convert Python object to JSON string and store in database"""
        if value is None:
            return None
        # Convert model/dictionary to JSON string
        return self.adapter.dump_json(value).decode('utf-8')

    def process_result_value(self, value, dialect):
        """Restore JSON string in database to Python object"""
        if value is None:
            return None
        # Parse JSON string back to corresponding Pydantic model structure
        return self.adapter.validate_json(value)

# ---------- Individual definition ----------
class PropertyRecord(SQLModel):
    """
    Record the score information for each property
    """
    score: float  # Score
    range: Tuple  # Score range
    for_llm: bool  # Whether scored by LLM
    evaluator: str  # e.g., "human", "GPT-4o", "simulator"


class Individual(SQLModel, table=True):
    """
    Individual information
    """
    # Index fields: Used for SQL retrieval and concurrency control
    id: Optional[int] = Field(default=None, primary_key=True)  # Primary key
    cell_index: str = Field(index=True)  # MAP-Elites coordinate index information
    island_id: int = Field(index=True)  # Island id
    is_elite: bool = Field(default=False, index=True)  # Whether is elite
    locked: bool = Field(default=False, index=True)  # Whether is being accessed
    is_removed: bool = Field(default=False, index=True)  # Whether is removed

    # Generation information
    parent_ids: List[int] = Field(default_factory=list, sa_column=Column(JSON))  # Parent id list
    generation: int  # Current generation
    origin: str  # Origin of individual, e.g., "init", "mutation", "crossover"
    iteration_found: int  # Record the iteration number when the individual is found

    # Core storage fields: Stored as JSON, not used for SQL indexing
    code: str
    fitness: List[float] = Field(sa_column=Column(JSON))  # Fitness value, support multi-objective optimization
    property_dict: Dict[str, PropertyRecord] = Field(
        default_factory=dict,
        sa_column=Column(PydanticJSON(Dict[str, PropertyRecord]))
    )  # Other properties to enhance diversity
    extraction_info: Dict[str, List[str]] = Field(
        default_factory=dict,
        sa_column=Column(JSON, name="reflection_info"),
    )  # Information returned by extractor; DB column name unchanged for existing DBs
    eval_info: Dict = Field(default_factory=dict, sa_column=Column(JSON))  # Evaluation information
    error_info: Dict = Field(default_factory=dict, sa_column=Column(JSON))  # Error information
