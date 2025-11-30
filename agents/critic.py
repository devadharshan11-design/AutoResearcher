from core.models import generate_text
from core.prompts import CRITIC_SYSTEM_PROMPT


def run_critic_agent(
    question: str,
    searcher_summary: str,
) -> str:
    """
    CRITIC agent:
    - Reviews the question and the searcher summary.
    - Points out missing aspects / issues.
    """
    user_prompt = (
        f"Question:\n{question}\n\n"
        f"Searcher agent summary:\n{searcher_summary}\n\n"
        "Now critique the summary as per your instructions."
    )

    critique = generate_text(
        system_prompt=CRITIC_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=0.2,
        max_tokens=400,
    )

    return critique
