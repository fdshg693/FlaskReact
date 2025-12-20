import asyncio
from dataclasses import dataclass

from agents import Agent, RunContextWrapper, Runner, function_tool


@dataclass
class Temperature:
    location: str
    celsius: float


@function_tool
async def get_temperature(wrapper: RunContextWrapper[Temperature]) -> str:
    """Get the current temperature. Call this function to get temperature information."""
    return (
        f"The temperature in {wrapper.context.location} is {wrapper.context.celsius}°C"
    )


async def main():
    # ロシアの温度コンテキスト
    russia_temp = Temperature(location="Russia", celsius=-20.0)

    # アフリカの温度コンテキスト
    africa_temp = Temperature(location="Africa", celsius=35.0)

    # ロシアAgent
    russia_agent = Agent[Temperature](
        name="Russian Assistant",
        tools=[get_temperature],
    )

    # アフリカAgent
    africa_agent = Agent[Temperature](
        name="African Assistant",
        tools=[get_temperature],
    )

    # ロシアAgentに温度を質問
    russia_result = await Runner.run(
        starting_agent=russia_agent,
        input="What is the temperature?",
        context=russia_temp,
    )

    # アフリカAgentに温度を質問
    africa_result = await Runner.run(
        starting_agent=africa_agent,
        input="What is the temperature?",
        context=africa_temp,
    )

    print("=== Russian Agent ===")
    print(russia_result.final_output)

    print("\n=== African Agent ===")
    print(africa_result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
