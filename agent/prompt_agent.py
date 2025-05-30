from world import WorldManager
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
        self,
        system_prompt_path: str,
        file_name: str,
        model: str,
        task_name: str,
        variation: int,
        method: str,
        master_name: str,
    ):
        self.man = WorldManager(
            file_name, task_name, variation, model, method, master_name
        )
        self.env = self.man.env
        self.model = OpenRouterProvider(
            model=model, api_key=os.getenv("OPENROUTER_API_KEY")
        )
        self.system_prompt = get_prompt(system_prompt_path)
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
            self.man.step_count += 1
            if self.man.step_count > 50:
                self.man.success = False
                self.man.finish_run()
                return True
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
            try:
                response = await self.model.generate(messages=history, temperature=0.1)
            except:
                print("Provider error. Retrying...")
                await asyncio.sleep(2)
                continue
            try:
                action = re.search(r">>> (.*?) <<", response.content).group(1)
                print(f"Agent action: {response.content}")
                self.context_store.store_string(response.content, role="assistant")
                self.man.add_step(self.man.step_count, response.content)
                break
            except AttributeError:
                print(f"Agent wrong format. Retrying...")
                continue
        self.current_step = self.env.step(action)
        self.man.score = self.current_step[3]["score"]
        if self.current_step[2]:
            print(f"Task completed!!!!!!!!!! Score: {self.current_step}")
            self.man.success = True
            self.man.finish_run()
            return True
        self.context_store.store_string(self.current_step[0], role="user")
        return False


tasks = [
    "boil",
    "change-the-state-of-matter-of",
    "chemistry-mix",
    "chemistry-mix-paint-secondary-color",
    "chemistry-mix-paint-tertiary-color",
    "find-animal",
    "find-living-thing",
    "find-non-living-thing",
    "find-plant",
    "freeze",
    "grow-fruit",
    "grow-plant",
    "identify-life-stages-1",
    "identify-life-stages-2",
    "inclined-plane-determine-angle",
    "inclined-plane-friction-named-surfaces",
    "inclined-plane-friction-unnamed-surfaces",
    "lifespan-longest-lived",
    "lifespan-longest-lived-then-shortest-lived",
    "lifespan-shortest-lived",
    "measure-melting-point-known-substance",
    "measure-melting-point-unknown-substance",
    "melt",
    "mendelian-genetics-known-plant",
    "mendelian-genetics-unknown-plant",
    "power-component",
    "power-component-renewable-vs-nonrenewable-energy",
    "test-conductivity",
    "test-conductivity-of-unknown-substances",
    "use-thermometer",
]


async def main():
    print(len(tasks))
    from llmgine.bootstrap import ApplicationBootstrap, ApplicationConfig

    app = ApplicationBootstrap(ApplicationConfig(enable_console_handler=False))
    await app.bootstrap()

    async def run_task(task):
        agent = PromptAgent(
            system_prompt_path="agent/prompts/system_prompt_v4.md",
            file_name=f"sonnet4-tips-{task}",
            model="anthropic/claude-sonnet-4",
            task_name=task,
            variation=0,
            method="v4",
            master_name="sonnet4-tips-default.csv",
        )
        while True:
            if await agent.step():
                break
        print(f"FINISHED task {task}!!!!!!!!!")
        agent.env.close()

    # await run_task("grow-plant")
    # await run_task("inclined-plane-determine-angle")
    # Process tasks in batches of 5
    for i in range(10, 30, 10):
        batch = tasks[i : i + 10]
        await asyncio.gather(*(run_task(task) for task in batch))
        print(f"Completed batch of {len(batch)} tasks")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
