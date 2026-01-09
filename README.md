## Quick Start Guide

1. Enter the LGU directory:
	```bash
	cd LGU
	```

2. Set up the environment:
	```bash
	conda env update -f environment.yaml
	conda activate lgu
	```

3. Generate answers:
	```bash
	./run_generate_answers.sh
	```

4. Compute LGU:
	```bash
	./compute_lgu.sh
	```

5. For experiment reproduction (OpenAI API required), you can directly use precomputed results in:
	```
	result/10_answer
	```

6. Analyze results:
	- Run notebooks in the `notebooks` folder for further analysis.
