from llm.langchain_custom import LLMModel, agent_run, create_agent_executor
from llm.langchain_custom.examples.sample_tools import add_numbers


def main() -> None:
    """Main function to execute the agent with the add_numbers tool."""
    agent = create_agent_executor(
        tools=[add_numbers],
        llm_model=LLMModel.GPT_4O_MINI,
    )

    prompt = "Add the numbers 1.5, 2.5, and 3.0 together."

    for response in agent_run(prompt, agent):
        print(response.pretty_print())


if __name__ == "__main__":
    from config import load_dotenv_workspace

    load_dotenv_workspace()
    main()
