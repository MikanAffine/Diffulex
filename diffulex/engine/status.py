from enum import Enum, auto


class DllmBlockType(Enum):
    """Type of a block during decoding."""
    IN_CONTEXT = auto()
    LAST_IN_CONTEXT = auto()
    OUT_OF_CONTEXT = auto()


class DllmBlockStatus(Enum):
    """State of a block during decoding."""
    DUMMY = auto()      # Placeholder block
    ACTIVE = auto()     # Currently being decoded
    TO_CACHE = auto()   # Ready to be cached
    IN_CACHE = auto()   # Already cached


class DllmReqStatus(Enum):
    """Status of a req during decoding."""
    # Waiting
    WAITING = auto()     # Waiting for prompts
    
    # Runtime states
    PENDING = auto()     # Pending to run
    PREFILLING = auto()  # Running prefilling
    DECODING = auto()    # Running decoding
    COMPLETED = auto()   # Completed generation
    
    # Finished
    FINISHED = auto()   # Completed generation