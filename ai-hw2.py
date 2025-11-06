from abc import ABC, abstractmethod

class State(ABC):
    @abstractmethod
    def __hash__(self):
        pass

    @abstractmethod
    def __eq__(self, other):
        pass

    @abstractmethod
    def __repr__(self):
        pass

class Problem(ABC):
    def __init__(self, initial_state):
        self.initial_state = initial_state

    @abstractmethod
    def goal_test(self, state):
        pass

    @abstractmethod
    def successors(self, state):
        pass

    def result(self, state, action):
        return action

class TowerOfHanoiState(State):
    def __init__(self, pegs):
        self.pegs = tuple(tuple(peg) for peg in pegs)

    def __hash__(self):
        return hash(self.pegs)

    def __eq__(self, other):
        return self.pegs == other.pegs

    def __repr__(self):
        return str(self.pegs)

class TowerOfHanoi(Problem):
    def __init__(self, num_disks):
        initial_state = TowerOfHanoiState((tuple(range(num_disks, 0, -1)), (), ()))
        super().__init__(initial_state)
        self.num_disks = num_disks
        self.goal_state = TowerOfHanoiState(((), (), tuple(range(num_disks, 0, -1))))

    def goal_test(self, state):
        return state == self.goal_state

    def successors(self, state):
        successors = []
        for from_peg in range(3):
            if not state.pegs[from_peg]:
                continue
            for to_peg in range(3):
                if from_peg == to_peg:
                    continue
                if state.pegs[to_peg] and state.pegs[to_peg][-1] < state.pegs[from_peg][-1]:
                    continue
                new_pegs = [list(peg) for peg in state.pegs]
                disk = new_pegs[from_peg].pop()
                new_pegs[to_peg].append(disk)
                new_state = TowerOfHanoiState(new_pegs)
                action = (from_peg, to_peg)
                successors.append((action, new_state))
        return successors