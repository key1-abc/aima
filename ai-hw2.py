from abc import ABC, abstractmethod
from typing import List, Tuple, Any, Optional

class State(ABC):
    @abstractmethod
    def __eq__(self, other: 'State') -> bool:
        pass
    
    @abstractmethod
    def __hash__(self) -> int:
        pass
    
    @abstractmethod
    def __str__(self) -> str:
        pass

class Problem(ABC):    
    @abstractmethod
    def get_initial_state(self) -> State:
        pass
    
    @abstractmethod
    def is_goal_state(self, state: State) -> bool:
        pass
    
    @abstractmethod
    def get_successors(self, state: State) -> List[Tuple[Any, State]]:
        pass
    
    @abstractmethod
    def get_goal_state(self) -> State:
        pass
    
    def get_cost(self, state1: State, action: Any, state2: State) -> float:
        return 1.0

class TowerOfHanoiState(State):
   
    def __init__(self, pegs: List[List[int]]):
        self.pegs = [peg.copy() for peg in pegs]  
        self.num_disks = sum(len(peg) for peg in pegs)
    
    def __eq__(self, other: 'TowerOfHanoiState') -> bool:
        if not isinstance(other, TowerOfHanoiState):
            return False
        return self.pegs == other.pegs
    
    def __hash__(self) -> int:
        return hash((tuple(tuple(peg) for peg in self.pegs)))
    
    def __str__(self) -> str:
        return f"TowerOfHanoiState(pegs={self.pegs})"
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def can_move(self, from_peg: int, to_peg: int) -> bool:
        if not (0 <= from_peg < 3 and 0 <= to_peg < 3):
            return False
        if not self.pegs[from_peg]:
            return False
        if not self.pegs[to_peg]:
            return True
        return self.pegs[from_peg][-1] < self.pegs[to_peg][-1]
    
    def move_disk(self, from_peg: int, to_peg: int) -> 'TowerOfHanoiState':
        if not self.can_move(from_peg, to_peg):
            raise ValueError("Invalid move")
        
        new_pegs = [peg.copy() for peg in self.pegs]
        disk = new_pegs[from_peg].pop()
        new_pegs[to_peg].append(disk)    
        return TowerOfHanoiState(new_pegs)

class TowerOfHanoiProblem(Problem):    
    def __init__(self, num_disks: int = 3):
        self.num_disks = num_disks
        initial_pegs = [list(range(num_disks, 0, -1)), [], []]
        self._initial_state = TowerOfHanoiState(initial_pegs)
        goal_pegs = [[], [], list(range(num_disks, 0, -1))]
        self._goal_state = TowerOfHanoiState(goal_pegs)
    
    def get_initial_state(self) -> TowerOfHanoiState:
        return self._initial_state
    
    def get_goal_state(self) -> TowerOfHanoiState:
        return self._goal_state
    
    def is_goal_state(self, state: TowerOfHanoiState) -> bool:
        return state == self._goal_state
    
    def get_successors(self, state: TowerOfHanoiState) -> List[Tuple[Tuple[int, int], TowerOfHanoiState]]:
        successors = []
        for from_peg in range(3):
            for to_peg in range(3):
                if from_peg != to_peg and state.can_move(from_peg, to_peg):
                    new_state = state.move_disk(from_peg, to_peg)
                    action = (from_peg, to_peg)
                    successors.append((action, new_state))
        return successors
    
    def heuristic(self, state: TowerOfHanoiState) -> float:
        disks_on_goal = len(state.pegs[2])
        return self.num_disks - disks_on_goal
