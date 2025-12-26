from abc import ABC, abstractmethod
from typing import List, Tuple, Any, Optional, Dict, Set, Deque
import heapq
import time
from collections import deque
import math

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
    
    def __lt__(self, other: 'State') -> bool:
        return hash(self) < hash(other)

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
        self._key = tuple(tuple(peg) for peg in self.pegs)
    
    def __eq__(self, other: 'TowerOfHanoiState') -> bool:
        if not isinstance(other, TowerOfHanoiState):
            return False
        return self._key == other._key
    
    def __hash__(self) -> int:
        return hash(self._key)
    
    def __str__(self) -> str:
        return f"Pegs: {self.pegs}"
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def __lt__(self, other: 'TowerOfHanoiState') -> bool:
        return hash(self) < hash(other)
    
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
    
    def heuristic_estimate(self, goal_pegs: List[List[int]]) -> float:
        disks_on_goal = len(self.pegs[2])
        total_disks = self.num_disks
        return total_disks - disks_on_goal

class TowerOfHanoiProblem(Problem):
    def __init__(self, num_disks: int = 3):
        self.num_disks = num_disks
        
        initial_pegs = [list(range(num_disks, 0, -1)), [], []]
        self._initial_state = TowerOfHanoiState(initial_pegs)
        
        goal_pegs = [[], [], list(range(num_disks, 0, -1))]
        self._goal_state = TowerOfHanoiState(goal_pegs)
        self._goal_pegs = goal_pegs
    
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
        return state.heuristic_estimate(self._goal_pegs)

class Node:
    def __init__(self, priority, state, path, g_cost=0):
        self.priority = priority
        self.state = state
        self.path = path
        self.g_cost = g_cost
    
    def __lt__(self, other):
        if self.priority != other.priority:
            return self.priority < other.priority
        return hash(self.state) < hash(other.state)

class SearchAlgorithm(ABC):
    def __init__(self, problem: Problem):
        self.problem = problem
        self.nodes_expanded = 0
        self.max_depth = 0
        self.start_time = time.time()
    
    @abstractmethod
    def search(self) -> Tuple[bool, List[Any], Dict[str, Any]]:
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            'nodes_expanded': self.nodes_expanded,
            'max_depth': self.max_depth,
            'running_time': time.time() - self.start_time
        }

class DFS(SearchAlgorithm):
    def __init__(self, problem: Problem, depth_limit: int = float('inf')):
        super().__init__(problem)
        self.depth_limit = depth_limit
    
    def search(self) -> Tuple[bool, List[Any], Dict[str, Any]]:
        initial_state = self.problem.get_initial_state()
        stack = [(initial_state, [], 0)]
        visited = set()
        
        while stack:
            current_state, path, depth = stack.pop()
            self.nodes_expanded += 1
            self.max_depth = max(self.max_depth, depth)
            
            if current_state in visited:
                continue
            visited.add(current_state)
            
            if self.problem.is_goal_state(current_state):
                return True, path, self.get_stats()
            
            if depth >= self.depth_limit:
                continue
            
            for action, next_state in self.problem.get_successors(current_state):
                if next_state not in visited:
                    stack.append((next_state, path + [action], depth + 1))
        
        return False, [], self.get_stats()

class BFS(SearchAlgorithm):
    def search(self) -> Tuple[bool, List[Any], Dict[str, Any]]:
        initial_state = self.problem.get_initial_state()
        queue = deque([(initial_state, [])])
        visited = set()
        
        while queue:
            current_state, path = queue.popleft()
            self.nodes_expanded += 1
            self.max_depth = max(self.max_depth, len(path))
            
            if current_state in visited:
                continue
            visited.add(current_state)
            
            if self.problem.is_goal_state(current_state):
                return True, path, self.get_stats()
            
            for action, next_state in self.problem.get_successors(current_state):
                if next_state not in visited:
                    queue.append((next_state, path + [action]))
        
        return False, [], self.get_stats()

class IDS(SearchAlgorithm):
    def search(self) -> Tuple[bool, List[Any], Dict[str, Any]]:
        initial_state = self.problem.get_initial_state()
        depth = 0
        
        while True:
            dfs = DFS(self.problem, depth)
            found, path, stats = dfs.search()
            
            self.nodes_expanded += dfs.nodes_expanded
            self.max_depth = max(self.max_depth, dfs.max_depth)
            
            if found:
                stats = self.get_stats()
                return True, path, stats
            
            if stats['running_time'] > 10:  
                return False, [], self.get_stats()
            
            depth += 1

class UCS(SearchAlgorithm):
    def search(self) -> Tuple[bool, List[Any], Dict[str, Any]]:
        initial_state = self.problem.get_initial_state()
        heap = [Node(0, initial_state, [])]
        visited = {initial_state: 0}
        
        while heap:
            node = heapq.heappop(heap)
            current_state, path, cost = node.state, node.path, node.priority
            self.nodes_expanded += 1
            self.max_depth = max(self.max_depth, len(path))
            
            if self.problem.is_goal_state(current_state):
                return True, path, self.get_stats()
            
            for action, next_state in self.problem.get_successors(current_state):
                new_cost = cost + self.problem.get_cost(current_state, action, next_state)
                
                if next_state not in visited or new_cost < visited[next_state]:
                    visited[next_state] = new_cost
                    heapq.heappush(heap, Node(new_cost, next_state, path + [action]))
        
        return False, [], self.get_stats()

class GreedyBestFirstSearch(SearchAlgorithm):
    def search(self) -> Tuple[bool, List[Any], Dict[str, Any]]:
        initial_state = self.problem.get_initial_state()
        heap = [Node(self.problem.heuristic(initial_state), initial_state, [])]
        visited = set()
        
        while heap:
            node = heapq.heappop(heap)
            current_state, path, heuristic_val = node.state, node.path, node.priority
            self.nodes_expanded += 1
            self.max_depth = max(self.max_depth, len(path))
            
            if current_state in visited:
                continue
            visited.add(current_state)
            
            if self.problem.is_goal_state(current_state):
                return True, path, self.get_stats()
            
            for action, next_state in self.problem.get_successors(current_state):
                if next_state not in visited:
                    h = self.problem.heuristic(next_state)
                    heapq.heappush(heap, Node(h, next_state, path + [action]))
        
        return False, [], self.get_stats()

class AStarSearch(SearchAlgorithm):
    def search(self) -> Tuple[bool, List[Any], Dict[str, Any]]:
        initial_state = self.problem.get_initial_state()
        
        heap = [Node(self.problem.heuristic(initial_state), initial_state, [], 0)]
        visited = {initial_state: 0}
        
        while heap:
            node = heapq.heappop(heap)
            current_state, path, g_cost = node.state, node.path, node.g_cost
            self.nodes_expanded += 1
            self.max_depth = max(self.max_depth, len(path))
            
            if self.problem.is_goal_state(current_state):
                return True, path, self.get_stats()
            
            for action, next_state in self.problem.get_successors(current_state):
                new_g = g_cost + self.problem.get_cost(current_state, action, next_state)
                new_h = self.problem.heuristic(next_state)
                new_f = new_g + new_h
                
                if next_state not in visited or new_g < visited[next_state]:
                    visited[next_state] = new_g
                    heapq.heappush(heap, Node(new_f, next_state, path + [action], new_g))
        
        return False, [], self.get_stats()

class HanoiSolver:
    def __init__(self, num_disks: int = 3):
        self.problem = TowerOfHanoiProblem(num_disks)
        self.num_disks = num_disks
    
    def solve_with_algorithm(self, algorithm_name: str) -> Tuple[bool, List[Any], Dict[str, Any]]:
        algorithms = {
            'DFS': DFS,
            'BFS': BFS,
            'IDS': IDS,
            'UCS': UCS,
            'GREEDY': GreedyBestFirstSearch,
            'ASTAR': AStarSearch
        }
        
        if algorithm_name not in algorithms:
            raise ValueError(f"Unknown algorithm: {algorithm_name}")
        
        algorithm = algorithms[algorithm_name](self.problem)
        return algorithm.search()
    
    def solve_all(self) -> Dict[str, Dict[str, Any]]:
        results = {}
        
        algorithms_to_test = ['BFS', 'DFS', 'IDS', 'UCS', 'GREEDY', 'ASTAR']
        
        for algo_name in algorithms_to_test:
            print(f"\n{'='*50}")
            print(f"Solving with {algo_name}...")
            print(f"{'='*50}")
            
            try:
                success, path, stats = self.solve_with_algorithm(algo_name)
                
                results[algo_name] = {
                    'success': success,
                    'path_length': len(path) if success else 0,
                    'path': path,
                    'stats': stats
                }
                
                self._print_results(algo_name, success, path, stats)
                
            except Exception as e:
                print(f"Error with {algo_name}: {e}")
                results[algo_name] = {
                    'success': False,
                    'path_length': 0,
                    'path': [],
                    'stats': {'nodes_expanded': 0, 'max_depth': 0, 'running_time': 0},
                    'error': str(e)
                }
        
        return results
    
    def _print_results(self, algorithm: str, success: bool, path: List, stats: Dict):
        print(f"\nAlgorithm: {algorithm}")
        print(f"Success: {success}")
        if success:
            print(f"Path length: {len(path)} moves (optimal: {2**self.num_disks - 1})")
        else:
            print(f"Path length: 0 moves")
        print(f"Nodes expanded: {stats['nodes_expanded']:,}")
        print(f"Max depth: {stats['max_depth']}")
        print(f"Running time: {stats['running_time']:.4f} seconds")
        
        if success and len(path) <= 15:  
            print("\nSolution path:")
            for i, (from_peg, to_peg) in enumerate(path, 1):
                print(f"  {i}. Move disk from peg {from_peg} to peg {to_peg}")
    
    def visualize_solution(self, path: List[Tuple[int, int]]):
        """Visualize the solution step by step"""
        state = self.problem.get_initial_state()
        print("\n" + "="*60)
        print("VISUALIZING SOLUTION")
        print("="*60)
        
        print(f"\nInitial state (step 0):")
        self._print_state(state)
        
        for step, (from_peg, to_peg) in enumerate(path, 1):
            print(f"\nStep {step}: Move disk from peg {from_peg} to peg {to_peg}")
            state = state.move_disk(from_peg, to_peg)
            self._print_state(state)
        
        print("\n✓ Solution completed!")
    
    def _print_state(self, state: TowerOfHanoiState):
        """Print current state visually"""
        max_height = self.num_disks
        
        print("\nCurrent configuration:")
        for level in range(max_height - 1, -1, -1):
            row = ""
            for peg in range(3):
                if level < len(state.pegs[peg]):
                    disk_size = state.pegs[peg][level]
                    disk_visual = "█" * (disk_size * 2 - 1)
                    row += f" {disk_visual:^{self.num_disks * 2}} "
                else:
                    row += f" {'│':^{self.num_disks * 2}} "
            print(row)
        
        base = "─" * (self.num_disks * 2 + 2)
        print(f"{base} {base} {base}")
        print(f"{'Peg 0':^{self.num_disks * 2 + 2}} {'Peg 1':^{self.num_disks * 2 + 2}} {'Peg 2':^{self.num_disks * 2 + 2}}")

def main():
    print("🚀 TOWER OF HANOI SOLVER")
    print("🎯 Implementing DFS, BFS, IDS, UCS, Greedy, and A* algorithms")
    
    while True:
        print("\n" + "="*60)
        print("MAIN MENU")
        print("="*60)
        print("1. Solve with specific algorithm")
        print("2. Compare all algorithms")
        print("3. Visualize solution")
        print("4. Exit")
        
        choice = input("\nSelect option (1-4): ").strip()
        
        if choice == '1':
            num_disks = int(input("Number of disks (default 3): ") or "3")
            if num_disks > 5:
                print("Warning: More than 5 disks may take a long time!")
                continue
            
            print("\nAvailable algorithms:")
            print("1. DFS (Depth-First Search)")
            print("2. BFS (Breadth-First Search)")
            print("3. IDS (Iterative Deepening Search)")
            print("4. UCS (Uniform Cost Search)")
            print("5. GREEDY (Greedy Best-First Search)")
            print("6. ASTAR (A* Search)")
            
            algo_choice = input("\nSelect algorithm (1-6): ").strip()
            algo_map = {'1': 'DFS', '2': 'BFS', '3': 'IDS', 
                       '4': 'UCS', '5': 'GREEDY', '6': 'ASTAR'}
            
            if algo_choice in algo_map:
                solver = HanoiSolver(num_disks)
                success, path, stats = solver.solve_with_algorithm(algo_map[algo_choice])
                
                solver._print_results(algo_map[algo_choice], success, path, stats)
                
                if success and input("\nVisualize solution? (y/n): ").lower() == 'y':
                    solver.visualize_solution(path)
            else:
                print("Invalid algorithm choice!")
        
        elif choice == '2':
            num_disks = int(input("Number of disks (1-4 recommended): ") or "3")
            if num_disks > 4:
                print("Warning: More than 4 disks may take a very long time!")
                continue
            
            solver = HanoiSolver(num_disks)
            results = solver.solve_all()
            
            print("\n" + "="*80)
            print("ALGORITHM COMPARISON SUMMARY")
            print("="*80)
            
            print(f"\n{'Algorithm':<10} {'Success':<10} {'Path Length':<12} {'Nodes Expanded':<15} {'Time (s)':<10}")
            print("-" * 60)
            
            for algo_name in ['BFS', 'DFS', 'IDS', 'UCS', 'GREEDY', 'ASTAR']:
                if algo_name in results:
                    result = results[algo_name]
                    stats = result['stats']
                    print(f"{algo_name:<10} {str(result['success']):<10} "
                          f"{result['path_length']:<12} {stats['nodes_expanded']:<15,} "
                          f"{stats['running_time']:<10.4f}")
        
        elif choice == '3':
            num_disks = int(input("Number of disks (default 3): ") or "3")
            if num_disks > 6:
                print("Too many disks for visualization!")
                continue
            
            solver = HanoiSolver(num_disks)
            
            print("\n1. Solve with BFS and visualize")
            print("2. Show optimal solution steps")
            vis_choice = input("\nSelect option (1-2): ").strip()
            
            if vis_choice == '1':
                success, path, stats = solver.solve_with_algorithm('BFS')
                if success:
                    solver.visualize_solution(path)
                else:
                    print("No solution found!")
            elif vis_choice == '2':
                success, path, stats = solver.solve_with_algorithm('BFS')
                if success:
                    solver.visualize_solution(path)
                else:
                    print("No solution found!")
        
        elif choice == '4':
            print("Goodbye!")
            break
        
        else:
            print("Invalid choice!")

if __name__ == "__main__":

    main()
