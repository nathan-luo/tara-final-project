from scienceworld import ScienceWorldEnv
from world import WorldManager

env: ScienceWorldEnv = None


def main():
    man = WorldManager("unlucky", "boil", 0, "gpt-4o", "test")
    env = man.env
    print(env.get_task_description())
    print(env.get_goal_progress())
    print(env.get_gold_action_sequence())
    print(env.step("open door to kitchen"))
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
    man.finish_run()


if __name__ == "__main__":
    main()
