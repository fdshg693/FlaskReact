from agents import Agent, Runner, RunResult

agent = Agent(
    name="Assistant", instructions="Reply in haiku format.", model="gpt-5-nano"
)

result: RunResult = Runner.run_sync(agent, "tell me a joke about computers")
print(result.final_output)
