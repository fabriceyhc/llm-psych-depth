python -m story_generation.generate \
          generator_args.model_name_or_path="TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ" \
          generation_args.strategy="writer_profile" \
          cuda_visible_devices=\'2,3\'

python -m story_generation.generate \
          generator_args.model_name_or_path="TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ" \
          generation_args.strategy="plan_write" \
          cuda_visible_devices=\'4,5\'

python -m story_generation.generate \
          generator_args.model_name_or_path="TheBloke/Llama-2-7B-chat-GPTQ" \
          generation_args.strategy="writer_profile" \
          cuda_visible_devices=\'0,1\'

python -m story_generation.generate \
          generator_args.model_name_or_path="TheBloke/Llama-2-7B-chat-GPTQ" \
          generation_args.strategy="plan_write" \
          cuda_visible_devices=\'0,1\'

python -m story_generation.generate \
          generator_args.model_name_or_path="TheBloke/Llama-2-13B-chat-GPTQ" \
          generation_args.strategy="writer_profile" \
          cuda_visible_devices=\'0,1\'

python -m story_generation.generate \
          generator_args.model_name_or_path="TheBloke/Llama-2-13B-chat-GPTQ" \
          generation_args.strategy="plan_write" \
          cuda_visible_devices=\'6,7\'

python -m story_generation.generate \
          generator_args.model_name_or_path="TheBloke/Llama-2-70B-chat-GPTQ" \
          generation_args.strategy="writer_profile" \
          cuda_visible_devices=\'0,1\'

python -m story_generation.generate \
          generator_args.model_name_or_path="TheBloke/Llama-2-70B-chat-GPTQ" \
          generation_args.strategy="plan_write" \
          cuda_visible_devices=\'0,1\'

python -m story_generation.generate \
          generator_args.model_name_or_path="lmsys/vicuna-33b-v1.3" \
          generation_args.strategy="writer_profile" \
          cuda_visible_devices=\'0,1,2,3\'

python -m story_generation.generate \
          generator_args.model_name_or_path="lmsys/vicuna-33b-v1.3" \
          generation_args.strategy="plan_write" \
          cuda_visible_devices=\'4,5,6,7\'

# python -m story_generation.generate \
#           generator_args.model_name_or_path="gpt-4" \
#           generation_args.strategy="writer_profile" 

# python -m story_generation.generate \
#           generator_args.model_name_or_path="gpt-4" \
#           generation_args.strategy="plan_write" 

# python -m story_generation.generate \
#           generator_args.model_name_or_path="gpt-3.5-turbo-0125" \
#           generation_args.strategy="writer_profile" 

# python -m story_generation.generate \
#           generator_args.model_name_or_path="gpt-3.5-turbo-0125" \
#           generation_args.strategy="plan_write" 