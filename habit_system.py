#!/usr/bin/env python3
"""
Sensor-Driven Habit-Formation Simulation System
Complete implementation with interactive experiment harness.

Author: NV
Date: 2025
"""

import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Any
import json
from datetime import datetime


# ============================================================================
# PART 1: SENSOR ENCODING AND MATRIX A CONSTRUCTION
# ============================================================================

class SensorPopulation:
    """Deterministic sensor population with M sensors."""
    
    def __init__(self, M: int, seed: int = 42):
        np.random.seed(seed)
        self.M = M
        self.seed = seed
        self.p = np.array([i % 4 for i in range(M)], dtype=float)
        self.g = np.random.uniform(0.8, 1.2, M)
        self.b = np.random.uniform(-0.1, 0.1, M)
        self.sigma = 0.6
        self.alpha = 1.0
    
    def compute_response(self, stimulus: int) -> np.ndarray:
        s = float(stimulus)
        distance_squared = (s - self.p) ** 2
        gaussian = np.exp(-distance_squared / (2 * self.sigma ** 2))
        pre_nonlinearity = self.g * gaussian + self.b
        response = self.alpha * np.log1p(np.exp(pre_nonlinearity))
        return response
    
    def build_matrix_A(self, stimuli: List[int]) -> np.ndarray:
        A = np.vstack([self.compute_response(s) for s in stimuli])
        return A


# ============================================================================
# PART 2: ARCHITECTURE PARAMETERS AND COMPUTE_X
# ============================================================================

class ArchitectureConfig:
    """Configuration for algebraic terms and parameters."""
    
    def __init__(self):
        self.params = {'a': 2, 'b': 3, 'c': 5}
        self.terms = [('a', 7, 1.0), ('b', 11, 0.8), ('c', 13, 0.6)]
        self.D = self.params['a'] * self.params['b'] * self.params['c']
    
    def compute_base(self, param_key: str, modulus: int) -> int:
        return self.params[param_key] % modulus
    
    def compute_x(self, A: np.ndarray, safe_mode: bool = True,
                   log_space: bool = False) -> np.ndarray:
        T, M = A.shape
        x = np.zeros_like(A)
        
        for param_key, modulus, weight in self.terms:
            base = self.compute_base(param_key, modulus)
            if safe_mode:
                base = max(1, min(base, 10))
            
            if log_space and base > 1:
                B_k = np.exp(A * np.log(base))
            else:
                B_k = base ** A
            
            x += weight * B_k
        
        x = x / self.D
        return x


# ============================================================================
# PART 3: FRACTIONAL KERNEL, STAGE-1 Y, AND ENERGY E
# ============================================================================

class FractionalKernel:
    """Compute fractional kernel and stage-1 y values."""
    
    @staticmethod
    def f(x: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
        x_safe = np.maximum(x, epsilon)
        sqrt_x = np.sqrt(x_safe)
        sqrt_4x = np.sqrt(4 * x_safe)
        numerator = np.mod(x_safe, sqrt_x)
        denominator = np.mod(x_safe, sqrt_4x) + epsilon
        result = numerator / denominator
        return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
    
    @staticmethod
    def compute_stage1_y(x: np.ndarray, e: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        f_x = FractionalKernel.f(x)
        if e is None:
            e = np.zeros_like(x)
        y = f_x + e
        return y, e


# ============================================================================
# PART 4: EXTENDED TRANSFORMS (X_S, LAMBDA, A0) AND STAGE-2 Y
# ============================================================================

class ExtendedTransforms:
    """Compute x_s, A0, and stage-2 y with lambda perturbation."""
    
    @staticmethod
    def compute_x_s(A: np.ndarray, e: np.ndarray, delta: float = 1e-8) -> np.ndarray:
        T, M = A.shape
        e_scaled = e / (10000 + delta)
        signs = np.zeros_like(A)
        for i in range(T):
            for j in range(M):
                signs[i, j] = (-1) ** (i + j)
        x_s = e_scaled * signs
        return x_s
    
    @staticmethod
    def compute_A0(A: np.ndarray, e: np.ndarray, lambda_interval: Tuple[float, float],
                    seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
        np.random.seed(seed)
        T, M = A.shape
        lambda_min, lambda_max = lambda_interval
        L = np.random.uniform(lambda_min, lambda_max, (T, M))
        A_lambda = A * L
        A0 = A + A_lambda
        return A0, L


# ============================================================================
# PART 5: Y_STORAGE - HISTORY, SYMBOLIC FORMS, AND REPRODUCIBILITY
# ============================================================================

class SymbolicETemplate:
    """Represent symbolic energy templates as tuples of coefficients."""
    
    def __init__(self, template_tuple: Tuple[float, ...]):
        self.coefficients = template_tuple
    
    def evaluate(self, numeric_values: np.ndarray) -> float:
        if len(numeric_values) == 0:
            return 0.0
        coeffs = np.array(self.coefficients[:len(numeric_values)])
        return float(np.dot(coeffs, numeric_values))
    
    def __repr__(self):
        return f"Template{self.coefficients}"


class SessionRecord:
    """Store a single session's data."""
    
    def __init__(self, inputs: List[int], A: np.ndarray, x: np.ndarray,
                 x_s: np.ndarray, L: np.ndarray, A0: np.ndarray,
                 y_stage1: np.ndarray, y_stage2: np.ndarray,
                 e_numeric: np.ndarray, e_symbolic: SymbolicETemplate,
                 timestamp: str = None, seed: int = None):
        self.inputs = inputs
        self.A = A
        self.x = x
        self.x_s = x_s
        self.L = L
        self.A0 = A0
        self.y_stage1 = y_stage1
        self.y_stage2 = y_stage2
        self.e_numeric = e_numeric
        self.e_symbolic = e_symbolic
        self.timestamp = timestamp or datetime.now().isoformat()
        self.seed = seed


class YStorage:
    """Append-only storage for session records with habit tracking."""
    
    def __init__(self):
        self.records: List[SessionRecord] = []
        self.habit_prototypes: Dict[str, Any] = {}
        self.habit_counter = 0
    
    def append_session(self, record: SessionRecord):
        self.records.append(record)
    
    def extract_numeric_e_collection(self) -> np.ndarray:
        all_e = []
        for record in self.records:
            all_e.append(record.e_numeric.flatten())
        return np.concatenate(all_e) if all_e else np.array([])
    
    def compute_habit_prototypes(self) -> Dict[str, np.ndarray]:
        e_collection = self.extract_numeric_e_collection()
        if len(e_collection) == 0:
            return {}
        prototypes = {
            'median_e': np.median(e_collection),
            'min_e': np.min(e_collection),
            'mean_e': np.mean(e_collection),
            'std_e': np.std(e_collection),
        }
        return prototypes
    
    def register_habit(self, habit_id: str, prototype_dict: Dict,
                       symbolic_template: SymbolicETemplate = None) -> str:
        habit_key = f"habit_{self.habit_counter}_{habit_id}"
        self.habit_prototypes[habit_key] = {
            'numeric': prototype_dict,
            'symbolic': symbolic_template,
        }
        self.habit_counter += 1
        return habit_key


# ============================================================================
# PART 6: HABIT EXTRACTION AND LABEL MAPPING
# ============================================================================

class HabitExtractor:
    """Extract and consolidate habit prototypes from storage."""
    
    @staticmethod
    def extract_habits_from_storage(storage: YStorage) -> Dict[str, Dict]:
        habits = {}
        e_collection = storage.extract_numeric_e_collection()
        if len(e_collection) > 0:
            habits['numeric_stats'] = {
                'median': float(np.median(e_collection)),
                'min': float(np.min(e_collection)),
                'max': float(np.max(e_collection)),
                'mean': float(np.mean(e_collection)),
                'std': float(np.std(e_collection)),
            }
        return habits


class LabelMapper:
    """Map recognized habits to input labels {0,1,2,3}."""
    
    @staticmethod
    def split_A0_into_label_groups(A0: np.ndarray, K: int = 4) -> List[np.ndarray]:
        A0_flat = A0.flatten()
        group_size = len(A0_flat) // K
        vectors = []
        for k in range(K):
            start_idx = k * group_size
            end_idx = len(A0_flat) if k == K - 1 else (k + 1) * group_size
            vectors.append(A0_flat[start_idx:end_idx])
        return vectors
    
    @staticmethod
    def compute_relevance_normalized_diff(element: float, prototype: float,
                                          tolerance: float = 0.1) -> float:
        normalized_diff = abs(element - prototype) / (abs(prototype) + 1e-8)
        return 1.0 if normalized_diff < tolerance else 0.0
    
    @staticmethod
    def map_habit_to_labels(A0: np.ndarray, habit_prototype: float,
                            label_groups: List[np.ndarray] = None,
                            tolerance: float = 0.15) -> Dict[str, Any]:
        if label_groups is None:
            label_groups = LabelMapper.split_A0_into_label_groups(A0, K=4)
        
        candidate_labels = []
        impulse_counts = [0, 0, 0, 0]
        relevance_scores = [0.0, 0.0, 0.0, 0.0]
        
        for label_k in range(len(label_groups)):
            v_k = label_groups[label_k]
            impulse_count = 0
            total_relevance = 0.0
            
            for i, element in enumerate(v_k):
                relevance = LabelMapper.compute_relevance_normalized_diff(
                    element, habit_prototype, tolerance
                )
                total_relevance += relevance
                impulse_count += int(relevance)
            
            impulse_counts[label_k] = impulse_count
            relevance_scores[label_k] = total_relevance / max(len(v_k), 1)
            
            if impulse_count > len(v_k) * 0.05:
                candidate_labels.append(label_k)
        
        return {
            'candidate_labels': candidate_labels,
            'impulse_counts': impulse_counts,
            'relevance_scores': relevance_scores,
        }


# ============================================================================
# PART 7: OUTPUT ALGORITHMS
# ============================================================================

class OutputAlgorithm:
    """Base class for output decision algorithms."""
    
    def decide(self, habit_info: Dict) -> str:
        raise NotImplementedError


class RuleBasedOutput(OutputAlgorithm):
    """Rule-based output algorithm."""
    
    def __init__(self, confidence_threshold: float = 0.1):
        self.confidence_threshold = confidence_threshold
    
    def decide(self, habit_info: Dict) -> str:
        candidates = habit_info.get('candidate_labels', [])
        relevance = habit_info.get('relevance_scores', [0]*4)
        impulses = habit_info.get('impulse_counts', [0]*4)
        
        if not candidates:
            return "UNCERTAIN (no matching habits)"
        
        best_label = max(candidates, key=lambda k: relevance[k])
        best_relevance = relevance[best_label]
        other_relevances = [relevance[k] for k in range(4) if k != best_label]
        margin = best_relevance - (max(other_relevances) if other_relevances else 0)
        confidence = "HIGH" if margin > self.confidence_threshold else "MEDIUM"
        
        return f"LABEL_{best_label} (confidence={confidence}, relevance={best_relevance:.3f})"
    
    def format_response(self, input_symbol: str, habit_info: Dict,
                        habit_id: str = None) -> str:
        decision = self.decide(habit_info)
        return f"Input: {input_symbol} | Habit: {habit_id or 'unknown'} | Decision: {decision}"


class EnhancedAnalysisOutput(OutputAlgorithm):
    """Enhanced output with multi-factor confidence."""
    
    def __init__(self, confidence_threshold: float = 0.1):
        self.confidence_threshold = confidence_threshold
    
    def decide(self, habit_info: Dict) -> str:
        candidates = habit_info.get('candidate_labels', [])
        relevance = habit_info.get('relevance_scores', [0]*4)
        impulses = habit_info.get('impulse_counts', [0]*4)
        
        if not candidates:
            return "UNCERTAIN (no matching habits)"
        
        relevance_array = np.array(relevance)
        impulse_array = np.array(impulses)
        max_impulses = np.max(impulse_array) if np.max(impulse_array) > 0 else 1
        impulse_norm = impulse_array / max_impulses
        combined = 0.6 * relevance_array + 0.4 * impulse_norm
        
        best_label = candidates[0]
        best_combined = combined[best_label]
        for label in candidates[1:]:
            if combined[label] > best_combined:
                best_label = label
                best_combined = combined[label]
        
        other_labels = [k for k in range(4) if k != best_label]
        other_scores = [combined[k] for k in other_labels]
        margin = best_combined - (max(other_scores) if other_scores else 0)
        
        if margin > self.confidence_threshold:
            confidence_level = "HIGH"
        elif best_combined > 0.1:
            confidence_level = "MEDIUM"
        else:
            confidence_level = "LOW"
        
        return (f"LABEL_{best_label} (confidence={confidence_level}, "
                f"relevance={relevance[best_label]:.3f}, impulse={impulses[best_label]}, "
                f"margin={margin:.3f})")
    
    def format_response(self, input_symbol: str, habit_info: Dict,
                        habit_id: str = None) -> str:
        decision = self.decide(habit_info)
        return f"Input: {input_symbol} | Habit: {habit_id or 'unknown'} | {decision}"


# ============================================================================
# PART 8: EXPERIMENT SYSTEM ORCHESTRATOR
# ============================================================================

class HabitFormationSystem:
    """Complete habit-formation simulation system."""
    
    def __init__(self, num_sensors: int = 8, output_algorithm: str = "rule_based",
                 seed: int = 42):
        np.random.seed(seed)
        self.seed = seed
        self.sensors = SensorPopulation(M=num_sensors, seed=seed)
        self.arch = ArchitectureConfig()
        self.kernel = FractionalKernel()
        self.transforms = ExtendedTransforms()
        self.storage = YStorage()
        
        if output_algorithm == "rule_based":
            self.output_algo = RuleBasedOutput(confidence_threshold=0.1)
        elif output_algorithm == "enhanced":
            self.output_algo = EnhancedAnalysisOutput(confidence_threshold=0.1)
        else:
            raise ValueError(f"Unknown algorithm: {output_algorithm}")
        
        self.output_algorithm_type = output_algorithm
        self.session_counter = 0
    
    def process_stimulus(self, stimulus: int, store_habit: bool = True) -> Dict[str, Any]:
        if stimulus not in {0, 1, 2, 3}:
            raise ValueError(f"Stimulus must be in {{0,1,2,3}}, got {stimulus}")
        
        A = self.sensors.build_matrix_A([stimulus])
        x = self.arch.compute_x(A, safe_mode=True)
        y_stage1, e_init = self.kernel.compute_stage1_y(x)
        x_s = self.transforms.compute_x_s(A, e_init)
        A0, L = self.transforms.compute_A0(A, e_init, (-1.4, -1.0), seed=self.seed)
        A0_safe = np.maximum(np.abs(A0), 1e-8)
        y_stage2 = x_s / A0_safe
        e_stage2 = y_stage2 - self.kernel.f(x)
        
        e_symbolic = SymbolicETemplate((1.0, 0.5, -0.2))
        record = SessionRecord(
            inputs=[stimulus], A=A, x=x, x_s=x_s, L=L, A0=A0,
            y_stage1=y_stage1, y_stage2=y_stage2,
            e_numeric=e_stage2, e_symbolic=e_symbolic, seed=self.seed
        )
        
        if store_habit:
            self.storage.append_session(record)
        
        habits = HabitExtractor.extract_habits_from_storage(self.storage)
        numeric_stats = habits.get('numeric_stats', {})
        label_groups = LabelMapper.split_A0_into_label_groups(A0, K=4)
        habit_prototype = numeric_stats.get('median', 0.0)
        mapping_result = LabelMapper.map_habit_to_labels(
            A0, habit_prototype, label_groups, tolerance=0.15
        )
        
        habit_id = f"habit_{self.session_counter}_{stimulus}"
        mapping_result['habit_prototype'] = habit_prototype
        mapping_result['habit_id'] = habit_id
        system_output = self.output_algo.format_response(
            str(stimulus), mapping_result, habit_id
        )
        
        self.session_counter += 1
        
        return {
            'stimulus': stimulus,
            'system_output': system_output,
            'num_sessions': len(self.storage.records),
        }
    
    def process_sequence(self, stimuli: List[int]) -> List[str]:
        outputs = []
        for stimulus in stimuli:
            result = self.process_stimulus(stimulus, store_habit=True)
            outputs.append(result['system_output'])
        return outputs


# ============================================================================
# PART 9: EXPERIMENT HARNESS
# ============================================================================

class ExperimentHarness:
    """Minimal test harness for running experiments."""
    
    def __init__(self, system: HabitFormationSystem, verbose: bool = True):
        self.system = system
        self.verbose = verbose
        self.experiment_log = []
    
    def run_single_input(self, input_symbol: str) -> str:
        if input_symbol not in {"0", "1", "2", "3"}:
            return f"ERROR: Invalid input '{input_symbol}'. Must be 0, 1, 2, or 3."
        
        stimulus = int(input_symbol)
        result = self.system.process_stimulus(stimulus, store_habit=True)
        output = result['system_output']
        self.experiment_log.append({
            'input': input_symbol,
            'stimulus': stimulus,
            'output': output,
            'timestamp': datetime.now().isoformat(),
        })
        return output
    
    def run_batch(self, stimuli: List[int], print_each: bool = True) -> List[str]:
        outputs = []
        for stimulus in stimuli:
            response = self.run_single_input(str(stimulus))
            if print_each:
                print(f"Input: {stimulus} → {response}")
            outputs.append(response)
        return outputs
    
    def print_summary(self):
        print("\n--- EXPERIMENT SUMMARY ---")
        print(f"Total inputs processed: {len(self.experiment_log)}")
        print(f"System sessions stored: {self.system.storage.habit_counter}")
        print(f"Algorithm: {self.system.output_algorithm_type}")
        
        if self.experiment_log:
            print("\nInput distribution:")
            for symbol in "0123":
                count = sum(1 for log in self.experiment_log if log['input'] == symbol)
                print(f"  Symbol {symbol}: {count} times")
        print()
    
    def run_interactive_loop(self, max_iterations: int = None):
        """Run interactive loop: read symbols and print responses."""
        iteration = 0
        print("\n" + "="*70)
        print("HABIT FORMATION SYSTEM - INTERACTIVE EXPERIMENT HARNESS")
        print("="*70)
        print("Enter symbols: 0, 1, 2, 3")
        print("Type 'quit' to exit | 'summary' for stats")
        print("="*70 + "\n")
        
        while max_iterations is None or iteration < max_iterations:
            try:
                user_input = input(">> ").strip()
                if not user_input:
                    continue
                if user_input.lower() in {'quit', 'exit', 'q'}:
                    print("Exiting...")
                    break
                if user_input.lower() == 'summary':
                    self.print_summary()
                    continue
                response = self.run_single_input(user_input)
                print(f"<< {response}\n")
                iteration += 1
            except KeyboardInterrupt:
                print("\n\nInterrupted.")
                break
            except Exception as e:
                print(f"Error: {e}\n")


# ============================================================================
# MAIN: USAGE
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("HABIT FORMATION SYSTEM - DEMO")
    print("="*80 + "\n")
    
    # Create system
    system = HabitFormationSystem(num_sensors=8, output_algorithm="enhanced", seed=42)
    harness = ExperimentHarness(system, verbose=True)
    
    # Run demo sequence
    print("Running demo sequence: 0, 1, 2, 3, 0, 1, 2, 3\n")
    test_sequence = [0, 1, 2, 3, 0, 1, 2, 3]
    outputs = harness.run_batch(test_sequence, print_each=True)
    harness.print_summary()
    
    # Option: run interactive loop
    print("\nTo run interactive mode, uncomment the line below:")
    print("# harness.run_interactive_loop()")
