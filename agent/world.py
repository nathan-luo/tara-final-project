import uuid
import json
from typing import Any, List, Dict
from scienceworld import ScienceWorldEnv


class WorldManager:
    def __init__(
        self,
        file_name: str,
        task_name: str,
        variation: int,
        model_name: str,
        method: str,
        master_name: str,
    ):
        self.file_name = file_name
        self.master_name = master_name
        self.env = ScienceWorldEnv(
            envStepLimit=999
        )  # We manually count and don't rely on the envStepLimit
        self.env.load(task_name, variation, generateGoldPath=True)
        self.content: Dict[int, List[Any]] = {}
        self.run_id = str(uuid.uuid4())
        self.step_count = 0
        self.success = False
        self.score = 0
        self.run: Dict[str, Any] = {
            "model": model_name,
            "run_id": self.run_id,
            "task_name": task_name,
            "variation": variation,
            "method": method,
            "num_steps": 0,
            "success": False,
            "score": 0,
            "gold_path_length": None,
        }
        if self.env.goldPathGenerated:
            self.run["gold_path_length"] = len(self.env.get_gold_action_sequence())

    def finish_run(self) -> None:
        self.env.store_run_history(0, str(self.run["method"]))
        self.env.save_run_histories(self.file_name)
        with open("results/" + self.file_name + ".json", "w") as f:
            f.write(json.dumps(self.content, indent=4))
        self.run["success"] = self.success
        self.run["num_steps"] = self.step_count
        self.run["score"] = self.score
        row = [
            self.file_name,
            self.run["model"],
            self.run["run_id"],
            self.run["task_name"],
            self.run["variation"],
            self.run["method"],
            self.run["num_steps"],
            self.run["success"],
            self.run["score"],
            self.run["gold_path_length"],
        ]
        with open(self.master_name, "a") as f:
            f.write(",".join(str(x) for x in row) + "\n")

    def add_step(self, step: int, payload: Any) -> None:
        if step not in self.content:
            self.content[step] = []
        self.content[step].append(payload)
