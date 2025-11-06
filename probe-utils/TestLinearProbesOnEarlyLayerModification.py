from TrainingRun import *
from TrainingRunBuilder import *
from DatasetGenerator import *

def main():
    builder = TrainingRunBuilder().use_basic_hf_dataset("NeelNanda/pile-10k", 4).use_SAELens_model_dataset("NeelNanda/pile-10k", "gpt2-small", [], [], 4)
    print(builder.build())

main()
