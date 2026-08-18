import os

from dotenv import load_dotenv
from openai import OpenAI


# Load variables from .env
load_dotenv()


class CustomerServiceBot:

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-3.6-flash"
    ):
        # Create Gemini client
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )

        # Store model name
        self.model = model

        # Create empty conversation history
        self.conversation_history = []

        # Add system prompt
        system_prompt = self._get_system_prompt()

        self.conversation_history.append({
            "role": "system",
            "content": system_prompt
        })


    def _get_system_prompt(self):

        return """
You are a helpful and professional customer service assistant
for ShopEasy, an e-commerce platform.

Your job is to help customers with:

1. Order status
2. Product information
3. Returns
4. Technical support
5. General questions

Be polite, clear, concise, and helpful.

If you do not know something, do not make up information.
Instead, clearly tell the customer that you do not have access
to that information.
"""


    def classify_intent(self, user_message: str) -> str:

        prompt = f"""
Classify the following customer message into exactly one
of these categories:

- order_status
- product_info
- returns
- technical_support
- general

Customer message:
{user_message}

Return only the category name.
"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_completion_tokens=50
        )

        intent = response.choices[0].message.content.strip()

        return intent


    def generate_response(
        self,
        user_message: str,
        intent: str | None = None
    ) -> str:

        # If intent is not provided, classify it
        if intent is None:
            intent = self.classify_intent(user_message)

            print(f"[Intent detected: {intent}]")

        # Add user message to conversation history
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })

        # Send complete conversation history to Gemini
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.conversation_history,
            temperature=0.7,
            max_completion_tokens=300
        )

        # Get assistant response
        assistant_message = response.choices[0].message.content

        # Save assistant response to history
        self.conversation_history.append({
            "role": "assistant",
            "content": assistant_message
        })

        return assistant_message


    def reset_conversation(self):

        self.conversation_history = [
            {
                "role": "system",
                "content": self._get_system_prompt()
            }
        ]

        print("[Conversation reset]")


    def get_conversation_summary(self) -> str:

        if len(self.conversation_history) <= 1:
            return "No conversation to summarize yet."

        summary_prompt = """
Please provide a brief summary of this customer service conversation.

Include:

1. Main customer concerns or questions
2. Information provided by the bot
3. Current status or next steps

Keep it concise in 2-3 sentences.
"""

        summary_messages = self.conversation_history + [
            {
                "role": "user",
                "content": summary_prompt
            }
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=summary_messages,
            temperature=0.3,
            max_completion_tokens=200
        )

        return response.choices[0].message.content


def main():

    # Get Gemini API key from .env
    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        print("Error: GOOGLE_API_KEY is not set in .env")
        return

    # Create chatbot
    bot = CustomerServiceBot(api_key)

    print("Customer Service Bot initialized!")
    print("Commands: quit, reset, summary")
    print()

    while True:

        user_input = input("You: ").strip()

        if not user_input:
            continue

        # Quit
        if user_input.lower() in ["quit", "exit"]:
            print("Thank you for using Customer Service Bot!")
            break

        # Reset conversation
        if user_input.lower() == "reset":
            bot.reset_conversation()
            continue

        # Conversation summary
        if user_input.lower() == "summary":
            print("\n--- Conversation Summary ---")
            print(bot.get_conversation_summary())
            print("----------------------------\n")
            continue

        # Generate response
        response = bot.generate_response(user_input)

        print(f"\nBot: {response}\n")


if __name__ == "__main__":
    main()