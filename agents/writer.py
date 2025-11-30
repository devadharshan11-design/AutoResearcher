from core.models import generate_text
from core.prompts import WRITER_SYSTEM_PROMPT


def run_writer_agent(
    question: str,
    searcher_summary: str,
    critic_feedback: str,
) -> str:
    """
    WRITER agent:
    - Generates the final structured answer.
    """
    user_prompt = (
        f"Question:\n{question}\n\n"
        f"Searcher agent summary:\n{searcher_summary}\n\n"
        f"Critic agent feedback:\n{critic_feedback}\n\n"
        "Now produce the final structured answer as per your instructions."
    )

    final_answer = generate_text(
        system_prompt=WRITER_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=0.25,
        max_tokens=900,
    )

    return final_answer
