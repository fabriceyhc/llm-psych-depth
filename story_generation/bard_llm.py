from typing import Any, List, Mapping, Optional
import requests

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from bardapi import Bard, SESSION_HEADERS




class BardLLM(LLM):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def _llm_type(self) -> str:
        return "bard"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        session = requests.Session()
        token = "cggBj8RevjfRDcbRnVhf6hyL9eOVwIAdB7vqNJGuTJ2WF0i7vbhIicVNLY87FeveG8EsCw."
        session.cookies.set("__Secure-1PSID", token)
        session.cookies.set( "__Secure-1PSIDCC", "ACA-OxOM5fsLVY49RsOveBzhVNRFOJuIKQt_ZIDUmy1ekZLIO8QDduiCGrtRTWx0gZ53pA44Uw")
        session.cookies.set("__Secure-1PSIDTS", "sidts-CjEBNiGH7gpY0RdJC-c1D33uC4eaSNbaEi0Yil7LXrPNslYchQ_3cUG-EW4IhL1sMjH8EAA")
        session.headers = SESSION_HEADERS
        bard = Bard(token=token, session=session)
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        return bard.get_answer(prompt)['content']


if __name__ == "__main__":

    llm = BardLLM()    
    prompt = \
"""
You are a seasoned writer who has won several accolades for your emotionally rich stories. When you write, you delve deep into the human psyche, pulling from the reservoir of universal experiences that every reader, regardless of their background, can connect to. Your writing is renowned for painting vivid emotional landscapes, making readers not just observe but truly feel the world of your characters. Every piece you produce aims to draw readers in, encouraging them to reflect on their own lives and emotions. Your stories are a complex tapestry of relationships, emotions, and conflicts, each more intricate than the last.

Now write a 500-word story on the following prompt:

A centuries old vampire gets really into video games because playing a character who can walk around in the sun is the closest thing they have to experiencing the day again in centuries.
"""
    output = llm(prompt)
    print(output)
