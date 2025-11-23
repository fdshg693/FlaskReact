from llm import agent_run, create_agent_executor, LLMModel, AgentPrompt

from llm.tools.others.sample_tools import add_numbers


def main() -> None:
    """Main function to execute the agent with the add_numbers tool."""
    tools = [add_numbers]
    agent = create_agent_executor(
        tools,
        llm_model=LLMModel.GPT_4O_MINI,
    )

    prompt_content = "Add the numbers 1.5, 2.5, and 3.0 together."
    prompt = AgentPrompt(content=prompt_content)

    for response in agent_run(prompt, agent):
        print(response.pretty_print())


if __name__ == "__main__":
    main()
