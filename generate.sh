CUDA_VISIBLE_DEVICES=0 python -m story_generation.generate \
                                 generator_args.model_name_or_path="TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ" \
                                 generation_args.strategy="writer_profile" \
                                 cuda_visible_devices="0"

CUDA_VISIBLE_DEVICES=1 python -m story_generation.generate \
                                 generator_args.model_name_or_path="TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ" \
                                 generation_args.strategy="plan_write" \
                                 cuda_visible_devices="1"

CUDA_VISIBLE_DEVICES=2 python -m story_generation.generate \
                                 generator_args.model_name_or_path="lmsys/vicuna-33b-v1.3" \
                                 generation_args.strategy="writer_profile" \
                                 cuda_visible_devices="2"

CUDA_VISIBLE_DEVICES=3 python -m story_generation.generate \
                                 generator_args.model_name_or_path="lmsys/vicuna-33b-v1.3" \
                                 generation_args.strategy="plan_write" \
                                 cuda_visible_devices="3"

CUDA_VISIBLE_DEVICES=2 python -m story_generation.generate \
                                 generator_args.model_name_or_path="meta-llama/Llama-2-7b-chat-hf" \
                                 generation_args.strategy="writer_profile" \
                                 cuda_visible_devices="2"

CUDA_VISIBLE_DEVICES=5 python -m story_generation.generate \
                                 generator_args.model_name_or_path="meta-llama/Llama-2-7b-chat-hf" \
                                 generation_args.strategy="plan_write" \
                                 cuda_visible_devices="5"

CUDA_VISIBLE_DEVICES=6 python -m story_generation.generate \
                                 generator_args.model_name_or_path="meta-llama/Llama-2-13b-chat-hf" \
                                 generation_args.strategy="writer_profile" \
                                 cuda_visible_devices="6"

CUDA_VISIBLE_DEVICES=3 python -m story_generation.generate \
                                 generator_args.model_name_or_path="meta-llama/Llama-2-13b-chat-hf" \
                                 generation_args.strategy="plan_write" \
                                 cuda_visible_devices="3"

CUDA_VISIBLE_DEVICES=2 python -m story_generation.generate \
                                 generator_args.model_name_or_path="meta-llama/Llama-2-70b-chat-hf" \
                                 generation_args.strategy="writer_profile" \
                                 cuda_visible_devices=2

CUDA_VISIBLE_DEVICES=5 python -m story_generation.generate \
                                 generator_args.model_name_or_path="meta-llama/Llama-2-70b-chat-hf" \
                                 generation_args.strategy="plan_write" \
                                 cuda_visible_devices=5