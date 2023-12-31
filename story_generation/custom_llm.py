import asyncio
import json
from typing import Any, List, Mapping, Optional

import websockets

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM

class CustomLLM(LLM):

    # this URI is generated by running `/data1/llm-psych-depth/text-generation-webui/start_linux.sh`
    # NOTE: it changes every time the oobabooga server is started unfortunately...
    # public UI url: https://e8d819964353b68e00.gradio.live
    URI = 'wss://stones-hitachi-letter-respective.trycloudflare.com/api/v1/stream'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def _llm_type(self) -> str:
        return "custom"

    async def _api_call(self, prompt: str) -> str:
        request = {
            'prompt': prompt,
            'mode': 'chat-instruct', 
            'max_new_tokens': 2048,
            'auto_max_new_tokens': False,
            'max_tokens_second': 0,
            'preset': 'None',
            'do_sample': True,
            'temperature': 1.53,
            'top_p': 0.64,
            'typical_p': 1,
            'epsilon_cutoff': 0,
            'eta_cutoff': 0,
            'tfs': 1,
            'top_a': 0,
            'repetition_penalty': 1.07,
            'additive_repetition_penalty': 0,
            'repetition_penalty_range': 0,
            'top_k': 33,
            'min_length': 0,
            'no_repeat_ngram_size': 0,
            'num_beams': 1,
            'penalty_alpha': 0,
            'length_penalty': 1,
            'early_stopping': False,
            'mirostat_mode': 0,
            'mirostat_tau': 5,
            'mirostat_eta': 0.1,
            'grammar_string': '',
            'guidance_scale': 1,
            'negative_prompt': '',
            'seed': -1,
            'add_bos_token': True,
            'truncation_length': 2048,
            'ban_eos_token': False,
            'custom_token_bans': '',
            'skip_special_tokens': True,
            'stopping_strings': []
        }

        async with websockets.connect(self.URI, ping_interval=None) as websocket:
            await websocket.send(json.dumps(request))
            aggregated_response = ""

            while True:
                incoming_data = await websocket.recv()
                incoming_data = json.loads(incoming_data)
                match incoming_data['event']:
                    case 'text_stream':
                        aggregated_response += incoming_data['text']
                    case 'stream_end':
                        return aggregated_response

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        response = asyncio.run(self._api_call(prompt))
        return response

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {}



if __name__ == '__main__':
    llm = CustomLLM()
    prompt = \
"""
You are a seasoned writer who has won several accolades for your emotionally rich stories. When you write, you delve deep into the human psyche, pulling from the reservoir of universal experiences that every reader, regardless of their background, can connect to. Your writing is renowned for painting vivid emotional landscapes, making readers not just observe but truly feel the world of your characters. Every piece you produce aims to draw readers in, encouraging them to reflect on their own lives and emotions. Your stories are a complex tapestry of relationships, emotions, and conflicts, each more intricate than the last.

Now write a 500-word story on the following prompt:

A centuries old vampire gets really into video games because playing a character who can walk around in the sun is the closest thing they have to experiencing the day again in centuries.
"""
    output = llm(prompt)
    print(output)