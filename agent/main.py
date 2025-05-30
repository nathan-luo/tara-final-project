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

load_dotenv(override=True)

env: ScienceWorldEnv = None


def setup_env(task_num: int = 13, variation: int = 0, simplifications: str = ""):
    global env
    env = ScienceWorldEnv(envStepLimit=10000)
    taskNames = env.get_task_names()
    task = taskNames[task_num]
    env.load(task, variation, simplifications)
    print("Starting Task " + str(task_num) + ": " + task)
    return env


def setup_env_from_task_name(
    task_name: str, variation: int = 0, simplifications: str = ""
):
    global env
    env = ScienceWorldEnv(envStepLimit=10000)
    env.load(task_name, variation, simplifications)
    print("Starting Task " + str(task_name))
    return env


def construct_prompt(
    step_status: str, allowed_actions: list[str], allowed_objects: list[str]
):
    prompt = f"""
    Current Status: {step_status}
    Allowed Actions: {allowed_actions}
    Allowed Objects: {allowed_objects}
    """
    return prompt


def take_action(action: str):
    """
    Takes an action in the environment. OBJ is replaced with a valid object. For example, if the action is "open OBJ", and the valid objects are ["door", "window"], then the action will be "open door" or "open window".
    """
    global env
    return action


class Ariel:
    def __init__(self):
        global env
        self.env = env
        self.engine_id = "main"
        self.session_id = "main"
        self.context_store = SimpleChatHistory(
            engine_id=self.engine_id, session_id=self.session_id
        )
        self.tool_manager = ToolManager(
            engine_id=self.engine_id, session_id=self.session_id
        )
        # self.model = OpenAIProvider(model="gpt-4.1", api_key=os.getenv("OPENAI_API_KEY"))
        self.model = OpenRouterProvider(
            model="deepseek/deepseek-r1-0528", api_key=os.getenv("OPENROUTER_API_KEY")
        )
        self.system_prompt = get_prompt("agent/prompts/system_prompt.md")
        self.context_store.set_system_prompt(
            self.system_prompt.format(task_desc=env.get_task_description())
        )

        # Start first step
        self.current_step = env.step(input_str="look around")
        self.context_store.store_string(self.current_step[0], role="user")

    async def step(self) -> bool:
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


async def main():
    from llmgine.bootstrap import ApplicationBootstrap, ApplicationConfig

    # env = setup_env(task_num=3, variation=0, simplifications="")
    env = setup_env_from_task_name("boil")
    app = ApplicationBootstrap(ApplicationConfig(enable_console_handler=False))
    await app.bootstrap()
    ariel = Ariel()
    while True:
        if await ariel.step():
            break
    print("FINISHED YAY!!!!!!!!!")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())

# def main():
#     env = setup_env(task_num=1, variation=0, simplifications="")
#     from pprint import pprint

#     # pprint(env.get_possible_actions())
#     # pprint(env.get_possible_objects())
#     pprint(env.step(input_str="look around")[0])
#     temp(env)
#     pprint(env.step(input_str="open kitchen door")[0])
#     temp(env)
#     pprint(env.step(input_str="go kitchen")[0])
#     temp(env)
#     # pprint(env.step(input_str="look around")[0])
#     # temp(env)
#     # pprint(env.step(input_str="inventory")[0])
#     # print(env.get_task_description())
#     # print(env.get_goal_progress())

# if __name__ == "__main__":
#     main()
