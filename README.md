# CoT-GRICE
This is a repository for the code and results from the study "Chain-of-Thought and GRICE: The effect of Zero-shot CoT prompting on implicature understanding". The scripts were coded in cursor with the assistance from its built-in AI.

The command line arguments that can be used for main_runner.py are:

--config: Used to specify the path.

--start_iteration: The position in the dataset that the user wishes to start the iteration from.

--o: Is used to specify the output directory.

The study's abstract:

This study investigates how zero-shot chain-of-thought (ZS-CoT) prompting affects the language model GPT-3.5-Turbo's ability to handle pragmatic tasks. The model’s performance is analyzed and evaluated when applied with the GRICE dataset, which is designed to test implicature retrieval and conversation reasoning. The experiment compares ZS-CoT prompting with neutral zero-shot prompting to identify the effect of the CoT method. The results show that ZS-CoT significantly improves the model's ability to understand conversations and draw inferences from context. In contrast, only a small, and in some cases negative, effect of ZS-CoT was observed on tasks involving implicature retrieval. These findings show that the ability to draw implicit inferences differs from the ability to understand conversations. Consequently, the results show that CoT prompting affects these abilities in different ways. The study contributes to the understanding of LLM's pragmatic competence and highlights the importance of adapting prompting strategies to the specific task.
