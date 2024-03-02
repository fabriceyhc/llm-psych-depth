python -m story_generation.generate \
          generator_args.model_name_or_path="TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ" \
          generation_args.strategy="writer_profile" \
          cuda_visible_devices=\'0,1\'

python -m story_generation.generate \
          generator_args.model_name_or_path="TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ" \
          generation_args.strategy="plan_write" \
          cuda_visible_devices=\'0,1\'

python -m story_generation.generate \
          generator_args.model_name_or_path="lmsys/vicuna-33b-v1.3" \
          generation_args.strategy="writer_profile" \
          cuda_visible_devices=\'0,1\'

python -m story_generation.generate \
          generator_args.model_name_or_path="lmsys/vicuna-33b-v1.3" \
          generation_args.strategy="plan_write" \
          cuda_visible_devices=\'0,1\'

python -m story_generation.generate \
          generator_args.model_name_or_path="meta-llama/Llama-2-7b-chat-hf" \
          generation_args.strategy="writer_profile" \
          cuda_visible_devices=\'0,1\'

python -m story_generation.generate \
          generator_args.model_name_or_path="meta-llama/Llama-2-7b-chat-hf" \
          generation_args.strategy="plan_write" \
          cuda_visible_devices=\'0,1\'

python -m story_generation.generate \
          generator_args.model_name_or_path="meta-llama/Llama-2-13b-chat-hf" \
          generation_args.strategy="writer_profile" \
          cuda_visible_devices=\'0,1\'

python -m story_generation.generate \
          generator_args.model_name_or_path="meta-llama/Llama-2-13b-chat-hf" \
          generation_args.strategy="plan_write" \
          cuda_visible_devices=\'0,1\'

python -m story_generation.generate \
          generator_args.model_name_or_path="meta-llama/Llama-2-70b-chat-hf" \
          generation_args.strategy="writer_profile" \
          cuda_visible_devices=\'0,1\'

python -m story_generation.generate \
          generator_args.model_name_or_path="meta-llama/Llama-2-70b-chat-hf" \
          generation_args.strategy="plan_write" \
          cuda_visible_devices=\'0,1\'

# python -m story_generation.generate \
#           generator_args.model_name_or_path="gpt-4" \
#           generation_args.strategy="writer_profile" 

# python -m story_generation.generate \
#           generator_args.model_name_or_path="gpt-4" \
#           generation_args.strategy="plan_write" 