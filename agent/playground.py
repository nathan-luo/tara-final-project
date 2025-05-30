from scienceworld import ScienceWorldEnv

env: ScienceWorldEnv = None


def setup_env(task_num: int = 13, variation: int = 0, simplifications: str = ""):
    global env
    env = ScienceWorldEnv(envStepLimit=10000)
    taskNames = env.get_task_names()
    task = taskNames[task_num]
    env.load(task, variation, simplifications, generateGoldPath=True)
    print("Starting Task " + str(task_num) + ": " + task)
    return env


def main():
    env = setup_env(task_num=1, variation=0, simplifications="")
    print(env.get_task_description())
    print(env.get_goal_progress())
    print(env.get_gold_action_sequence())
    print(env.step("open door to kitchen")[0:3])
    print(env.step("go kitchen")[0:3])
    print(env.step("pick up cup")[0:3])
    print(env.step("use cup on sink")[0:3])
    print(env.step("dunk cup in sink")[0:3])
    print(env.step("activate sink")[0:3])
    print(env.step("move cup to sink")[0:3])
    print(env.step("focus on substance in cup")[0:3])
    print(env.step("activate stove")[0:3])
    print(env.step("move cup to stove")[0:3])
    print(env.step("wait")[0:3])
    print(env.step("focus on substance in cup")[0:3])
    print(env.step("wait")[0:3])
    print(env.get_goal_progress())
    print(env.store_run_history(1, "test2"))
    print(env.save_run_histories("test"))
    env.close()


# Starting Task boil
# --------------------------------
# This room is called the hallway. In it, you see:
#         the agent
#         a substance called air
#         a picture
# You also see:
#         A door to the art studio (that is closed)
#         A door to the bedroom (that is closed)
#         A door to the greenhouse (that is closed)
#         A door to the kitchen (that is closed)
#         A door to the living room (that is closed)
#         A door to the workshop (that is closed)
# --------------------------------
# Agent action: open door to kitchen
# --------------------------------
# The door is now open.
# --------------------------------
# Agent action: go kitchen
# --------------------------------
# You move to the kitchen.
# --------------------------------
# Agent action: pick up cup
# --------------------------------
# You move the glass cup to the inventory.
# --------------------------------
# Agent action: use cup on sink
# --------------------------------
# I'm not sure how to use the glass cup.
# --------------------------------
# Agent action: dunk cup in sink
# --------------------------------
# The sink does not contain any liquids to dunk into.
# --------------------------------
# Agent action: activate sink
# --------------------------------
# The sink is now activated.
# --------------------------------
# Agent action: move cup to sink
# --------------------------------
# You move the glass cup to the sink.
# --------------------------------
# Agent action: focus on substance in cup
# --------------------------------
# You focus on the water.
# --------------------------------
# Agent action: activate stove
# --------------------------------
# The stove is now activated.
# --------------------------------
# Agent action: move cup to stove
# --------------------------------
# You move the glass cup to the stove.
# --------------------------------
# Agent action: wait
# --------------------------------
# You decide to wait for 10 iterations.
# --------------------------------
# Agent action: focus on substance in cup
# --------------------------------
# You focus on the water.
# --------------------------------
# Agent action: wait
# Task completed!!!!!!!!!! Score: 25
# FINISHED YAY!!!!!!!!!
if __name__ == "__main__":
    main()
