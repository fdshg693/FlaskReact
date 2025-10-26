from llm import agent_run, LLMModel, ModelProvider, AgentPrompt

from llm.tools.others.sample_tools import add_numbers


def main() -> None:
    """Main function to execute the agent with the add_numbers tool."""
    tools = [add_numbers]
    prompt_content = "Add the numbers 1.5, 2.5, and 3.0 together."
    prompt = AgentPrompt(content=prompt_content)

    for response in agent_run(
        prompt,
        tools,
        LLMModel.GPT_4O_MINI,
        ModelProvider.OPENAI,
    ):
        print(response.pretty_print())


if __name__ == "__main__":
    main()
