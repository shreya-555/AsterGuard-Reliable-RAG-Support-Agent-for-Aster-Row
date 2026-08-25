import os

from openai import OpenAI

from app.config import MODEL_NAME


class GroqLLM:
    """
    LLM adapter using Groq's OpenAI-compatible API.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ):
        self.api_key = (
            api_key
            or os.getenv("GROQ_API_KEY")
        )

        self.model = (
            model
            or MODEL_NAME
        )

        if not self.api_key:
            raise ValueError(
                "GROQ_API_KEY is not configured."
            )

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=(
                "https://api.groq.com/openai/v1"
            ),
        )

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:

        try:
            response = (
                self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": system_prompt,
                        },
                        {
                            "role": "user",
                            "content": user_prompt,
                        },
                    ],
                    temperature=0.1,
                )
            )

            text = (
                response
                .choices[0]
                .message
                .content
            )

            if not text:
                raise RuntimeError(
                    "Model returned an empty response."
                )

            return text.strip()

        except Exception as exc:
            raise RuntimeError(
                f"LLM generation failed: {exc}"
            ) from exc