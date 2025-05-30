from scienceworld import ScienceWorldEnv
from llmgine.llm.context.memory import SimpleChatHistory
from llmgine.llm.tools.tool_manager import ToolManager
from llmgine.llm.models.openai_models import Gpt41, Gpt41Mini
from llmgine.llm.providers.providers import Providers
from llmgine.prompts.prompts import get_prompt
from llmgine.llm.providers.openai_provider import OpenAIProvider
from llmgine.llm.providers.openrouter import OpenRouterProvider
import re
import copy
import os
from dotenv import load_dotenv


class PromptAgent:
    def __init__(
        self, system_prompt: str, model: str, environment: ScienceWorldEnv, writer: Writer
    ):
        self.env = environment
        self.model = OpenRouterProvider(
            model=model, api_key=os.getenv("OPENROUTER_API_KEY")
        )
        self.system_prompt = get_prompt(system_prompt)
        self.context_store = SimpleChatHistory(engine_id="main", session_id="main")
        self.context_store.set_system_prompt(
            self.system_prompt.format(task_desc=self.env.get_task_description())
        )

        # Start first step
        self.current_step = self.env.step(input_str="look around")
        self.context_store.store_string(self.current_step[0], role="user")

    async def step(self):
        history = await self.context_store.retrieve()
        #  Think about what to do
        history = copy.deepcopy(history)
        print("--------------------------------")
        print(history[-1]["content"])
        print("--------------------------------")
        if history[-1]["content"].startswith("Ambiguous request:"):
            history[-1]["content"] = (
                history[-1]["content"]
                + "\n\n"
                + "Type the number corresponding to the action you want to take, like >>> num <<<"
            )
        else:
            history[-1]["content"] = (
                history[-1]["content"]
                + "\n\n"
                + "Your available actions are: "
                + str(self.env.get_possible_actions())
                + "\n\n"
                + "Your available objects are: "
                + str(self.env.get_possible_objects())
                + "\n\n"
                + f"Remember, your task is to {self.env.get_task_description()}"
                + "\n\n"
                + "End your response with the action you take, like >>> action <<<"
            )
        while True:
            response = await self.model.generate(messages=history)
            self.context_store.store_string(response.content, role="assistant")
            try:
                action = re.search(r">>> (.*?) <<", response.content).group(1)
                print(f"Agent action: {action}")
                break
            except AttributeError:
                print(f"Agent wrong format. Retrying...")
                continue
        self.current_step = self.env.step(action)
        if self.current_step[2]:
            print(f"Task completed!!!!!!!!!! Score: {self.current_step}")
            return True
        self.context_store.store_string(self.current_step[0], role="user")
        return False
