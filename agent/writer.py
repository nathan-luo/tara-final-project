import uuid
from scienceworld import ScienceWorldEnv


class WorldManager:
    def __init__(
        self,
    ):
        self.file_path = file_path
        self.env = env
        self.content = []

    def get_env(self):
        return self.env

    def start_run(
        self,
        model_name: str,
        description: str,
        task_name: str,
    ):
        self.run = {
            "model": model_name,
            "run_id": str(uuid.uuid4()),
            "chat_history": None,
            "description": None,
            "num_steps": 0,
            "score": 0,
            "gold_path": None,
            "gold_path_length": 0,
        }
        self.run_id = str(uuid.uuid4())

    def env_record_run(self, run_history: list[str]):
        self.env.store_run_history(self.run_id, run_history)

    def env_save_history(self, history: list[str]):
        self.env.save_run_histories()

    def dump_chat_history(self, history: list):
        self.append
