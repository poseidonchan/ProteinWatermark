import numpy as np
from abc import ABC, abstractmethod
from typing import Union

class AbstractWatermarkCode(ABC):
    @classmethod
    @abstractmethod
    def from_random(cls,
                    rng: Union[np.random.Generator, list[np.random.Generator]],
                    vocab_size: int):
        pass

class AbstractReweight(ABC):
    watermark_code_type: type[AbstractWatermarkCode]

    @abstractmethod
    def reweight(self,
                 code: AbstractWatermarkCode,
                 p: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def get_la_score(self,
                     code: AbstractWatermarkCode) -> np.ndarray:
        pass

def get_gumbel_variables(rng: np.random.Generator,
                         vocab_size: int):
    u = rng.random((vocab_size,))  # ~ Unifom(0, 1)
    e = -np.log(u)  # ~ Exp(1)
    g = -np.log(e)  # ~ Gumbel(0, 1)
    return u, e, g


class DeltaGumbel_WatermarkCode(AbstractWatermarkCode):
    def __init__(self, g: np.ndarray):
        self.g = g

    @classmethod
    def from_random(
            cls,
            rng: Union[np.random.Generator, list[np.random.Generator]],
            vocab_size: int,
    ):
        if isinstance(rng, list):
            batch_size = len(rng)
            g = np.stack(
                [get_gumbel_variables(rng[i], vocab_size)[2] for i in range(batch_size)]
            )
        else:
            g = get_gumbel_variables(rng, vocab_size)[2]

        return cls(g)


class DeltaGumbel_Reweight(AbstractReweight):
    watermark_code_type = DeltaGumbel_WatermarkCode

    def __repr__(self):
        return f"DeltaGumbel_Reweight()"

    def reweight(
            self, code: AbstractWatermarkCode, p_logits: np.ndarray
    ) -> np.ndarray:
        assert isinstance(code, DeltaGumbel_WatermarkCode)

        index = np.argmax(p_logits + code.g, axis=-1)

        mask = np.arange(p_logits.shape[-1]) == index[..., None]

        modified_logits = np.where(
            mask,
            np.full_like(p_logits, 0),
            np.full_like(p_logits, float("-inf")),
        )
        return modified_logits

    def get_la_score(self, code):
        """likelihood agnostic score"""
        return np.array(np.log(2)) - np.exp(-code.g)

class WatermarkDetector:
    def __init__(
            self,
            private_key: any,
            reweight: AbstractReweight,
            context_code_length: int,
            vocab_size: int = 20,
            ignore_history: bool = False
    ):
        self.private_key = private_key
        self.cc_length = context_code_length
        self.vocab_size = vocab_size
        self.reweight = reweight
        self.ignore_history = ignore_history
        self.cc_history = set()

    def reset_history(self):
        self.cc_history = set()

    def get_rng_seed(self, context_code: any) -> any:
        if not self.ignore_history:
            self.cc_history.add(context_code)
        import hashlib

        m = hashlib.sha256()
        m.update(context_code)
        m.update(self.private_key)
        full_hash = m.digest()
        seed = int.from_bytes(full_hash, "big") % (2 ** 32 - 1)
        return seed

    def _get_codes(self, context):
        batch_size = len(context)

        context_codes = [
            context[i][-self.cc_length:].tobytes() for i in range(batch_size)
        ]
        # print(context[0][-self.cc_length:])
        mask, seeds = zip(
            *[
                (context_code in self.cc_history, self.get_rng_seed(context_code))
                for context_code in context_codes
            ]
        )
        return mask, seeds

    def detect(self,
               input_ids: np.ndarray,
               method='original'):
        """
        :param input_ids: sequences after tokenization
        :return: scores, a higher score means a seq is likely to be watermarked
        """
        if method == 'optimized':
            scores = []
            for i in range(input_ids.shape[1]):
                score = self.get_la_score(input_ids[:, :i], input_ids[:, i], self.vocab_size)
                ti = score-np.log(2)
                scores.append(ti)
            assert np.all(ti<=0), ti
            tis = np.array(scores)
            uis = np.exp(tis)
            Ubar=np.mean(uis)
            # print("Ubar", Ubar)

            if np.mean(uis)==0:
                final_score=0
                final_p_value=1
                return final_score, final_p_value
            avgS = lambda Ubar, lamb: Ubar*lamb+np.log(lamb/np.expm1(lamb))
            import scipy.optimize
            sol=scipy.optimize.minimize(lambda l:-avgS(Ubar, l), 0.5, method='SLSQP', bounds=[(0,10)])
            final_score=-sol.fun*input_ids.shape[1]
            final_p_value = np.exp(-final_score)
            return final_score, final_p_value
        
        elif method == 'original':
            scores = []
            for i in range(input_ids.shape[1]):
                score = self.get_la_score(input_ids[:, :i], input_ids[:, i], self.vocab_size)
                scores.append(score)
            scores = np.array(scores)
            final_score = np.sum(scores, axis=0)
            return final_score, np.exp(-final_score)

    def get_la_score(
            self,
            input_ids: np.ndarray,
            labels: np.ndarray,
            vocab_size: int,
    ) -> np.ndarray:
        assert "get_la_score" in dir(
            self.reweight
        ), "Reweight does not support likelihood agnostic detection"
        mask, seeds = self._get_codes(input_ids)
        rng = [
            np.random.default_rng(seed) for seed in seeds
        ]
        
        mask = np.array(mask)
        watermark_code = self.reweight.watermark_code_type.from_random(rng, vocab_size)
        all_scores = self.reweight.get_la_score(watermark_code)
        scores = all_scores[np.arange(all_scores.shape[0]), labels]
        
        scores = np.logical_not(mask) * scores
        return scores


amino_acids = 'ACDEFGHIKLMNPQRSTVWYX'  # List of amino acids
# Create a dictionary mapping each amino acid to its index
aa_to_index = {aa: idx for idx, aa in enumerate(amino_acids)}
class AminoAcidTokenizer:
    def __init__(self, aa_to_index):
        self.aa_to_index = aa_to_index
        self.index_to_aa = {idx: aa for aa, idx in aa_to_index.items()}
        
    def encode(self, sequence):
        # Encode a sequence of amino acids to indices
        return np.array([self.aa_to_index.get(aa, self.aa_to_index.get('X')) for aa in sequence]).reshape(1,-1)
        
    def decode(self, indices):
        # Decode a list of indices back into an amino acid sequence
        return ''.join(self.index_to_aa.get(idx, 'X') for idx in indices)
tokenizer = AminoAcidTokenizer(aa_to_index)


import os
import csv
import argparse
from tqdm import tqdm

# Set up argument parsing
parser = argparse.ArgumentParser(description='Process a range of keys.')
parser.add_argument('--start_key', type=int, required=True, help='Starting key (inclusive)')
parser.add_argument('--end_key', type=int, required=True, help='Ending key (inclusive)')
args = parser.parse_args()

start_key = args.start_key
end_key = args.end_key

# Path to the design_outputs directory
design_outputs_dir = './design_outputs'

# Get a sorted list of all key directories
key_dirs = sorted([d for d in os.listdir(design_outputs_dir) if d.startswith('key_')])

# Extract all keys
all_keys = [d.split('_')[1] for d in key_dirs]

# Filter key_dirs to only include keys in the specified range
key_dirs = [d for d in key_dirs if start_key <= int(d.split('_')[1]) <= end_key]


# Read all sequences and their corresponding true_key
seq_data = []  # List of tuples (seq, true_key)
for key_dir in key_dirs:
    true_key = key_dir.split('_')[1]
    # Path to the 7YBX.fa file in the current key directory
    fa_file_path = os.path.join(design_outputs_dir, key_dir, 'seqs', '7YBX.fa')

    # Read the sequences from the .fa file
    with open(fa_file_path, 'r') as fa_file:
        lines = fa_file.readlines()
        # Sequences start from line 3
        sequence_lines = lines[2:]

        # Extract sequences (every second line starting from line 3)
        sequences = [line.strip() for idx, line in enumerate(sequence_lines) if idx % 2 == 1]

    for seq in sequences:
        seq_data.append((seq, true_key))

# Initialize the results list
results = []

# Cache detectors to avoid re-initializing for the same detect_key
detector_cache = {}

# Total number of tasks
total_tasks = len(seq_data) * len(all_keys)

# Progress bar to track processing
progress_bar = tqdm(total=total_tasks, desc="Processing tasks")

# Process each task separately
for seq, true_key in seq_data:
    # Encode the sequence
    encoded_seq = tokenizer.encode(seq)
    for detect_key in all_keys:
        # Initialize the watermark detector with the current detect_key
        detector = WatermarkDetector(
            bytes(detect_key, 'utf-8'),
            DeltaGumbel_Reweight(),
            context_code_length=5,
            vocab_size=21
        )

        # Detect the watermark score
        detection_result = detector.detect(encoded_seq, method='optimized')[0]
        watermark_score = detection_result / len(seq)

        # Append the result
        results.append({
            'seq': seq,
            'true_key': true_key,
            'detect_key': detect_key,
            'watermark_score': watermark_score
        })

        # Update progress bar
        progress_bar.update(1)

progress_bar.close()

# Write the results to a CSV file
output_file = f'./multikeys_results_{start_key}_{end_key}.csv'
with open(output_file, 'w', newline='') as csvfile:
    fieldnames = ['seq', 'true_key', 'detect_key', 'watermark_score']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for row in results:
        writer.writerow(row)

print(f'Results saved to {output_file}')
