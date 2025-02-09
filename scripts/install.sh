conda create -n oat-llm python==3.9.0 -y

conda activate oat-llm

pip install vllm==0.6.2 && pip install -e .
