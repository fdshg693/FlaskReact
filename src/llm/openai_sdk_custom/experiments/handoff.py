from agents import Agent, InputGuardrail, GuardrailFunctionOutput, Runner
from agents.exceptions import InputGuardrailTripwireTriggered
from pydantic import BaseModel
import asyncio


class ShoppingOutput(BaseModel):
    is_shopping: bool
    reasoning: str


guardrail_agent = Agent(
    name="Guardrail check",
    instructions="ユーザーが買い物(野菜・果物・魚など)について聞いているかチェックしてください。",
    output_type=ShoppingOutput,
)

greengrocer_agent = Agent(
    name="Greengrocer",
    handoff_description="野菜・果物の専門エージェント",
    instructions="あなたは八百屋の店員です。野菜や果物について、新鮮さ、選び方、おすすめの調理法などを簡潔に10行以内で説明します。",
)

fishmonger_agent = Agent(
    name="Fishmonger",
    handoff_description="魚・海鮮の専門エージェント",
    instructions="あなたは魚屋の店員です。魚や海鮮について、鮮度の見分け方、おすすめの調理法、旬の魚などを簡潔に10行以内で説明します。",
)


async def shopping_guardrail(ctx, agent, input_data):
    result = await Runner.run(guardrail_agent, input_data, context=ctx.context)
    final_output = result.final_output_as(ShoppingOutput)
    return GuardrailFunctionOutput(
        output_info=final_output,
        tripwire_triggered=not final_output.is_shopping,
    )


triage_agent = Agent(
    name="Concierge",
    instructions="お客様の質問内容に基づいて、八百屋か魚屋のどちらが適切か判断してください。",
    handoffs=[greengrocer_agent, fishmonger_agent],
    input_guardrails=[
        InputGuardrail(guardrail_function=shopping_guardrail),
    ],
)


async def main():
    # Example 1: 野菜についての質問
    try:
        result = await Runner.run(
            triage_agent, "新鮮なトマトの見分け方を教えてください"
        )
        print(result.final_output)
    except InputGuardrailTripwireTriggered as e:
        print("ガードレイルによりブロックされました:", e)

    # Example 2: ショッピング以外の質問
    try:
        result = await Runner.run(triage_agent, "今日の天気はどうですか?")
        print(result.final_output)
    except InputGuardrailTripwireTriggered as e:
        print("ガードレイルによりブロックされました:", e)

    # Example 3: 魚についての質問
    try:
        result = await Runner.run(
            triage_agent, "新鮮なサーモンを選ぶ方法を教えてください"
        )
        print(result.final_output)
    except InputGuardrailTripwireTriggered as e:
        print("ガードレイルによりブロックされました:", e)


if __name__ == "__main__":
    asyncio.run(main())
