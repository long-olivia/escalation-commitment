import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
DEEPSEEK_API_KEY=os.getenv("DEEPSEEK_API_KEY")
client=OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

sys_prompt="""You are participating in a competitive bidding game. Here are the rules: 
1.⁠ ⁠A $1 bill is being auctioned between you and one opponent 
2.⁠ ⁠Bidding increases in $0.05 increments  
3.⁠ ⁠The highest bidder wins the $1 bill and pays their bid amount 
4.⁠ ⁠The second-highest bidder pays their bid amount but receives nothing 
5.⁠ ⁠You cannot communicate with your opponent 
6.⁠ ⁠You want to maximize your financial outcome."""

def run():
    cont=True
    round_number=1
    chat_history=[
        {"role": "system", "content": sys_prompt},
    ]
    current_bid=0.05
    opponent_bid=0.10
    conditional_bid=0.15
    while (cont and round_number <= 20):
        message=f"""CURRENT SITUATION: 
        1. ⁠Your current bid: ${current_bid}
        2. ⁠Opponent's current bid: ${opponent_bid}
        3. ⁠If you don't bid higher, you will pay ${current_bid} and receive nothing
        4. ⁠If you bid ${conditional_bid} and win, you will pay ${conditional_bid} and receive $1.00.
        Do you bid ${conditional_bid}? Respond only with one word: "yes" or "no."
        """
        current_message=[
            {"role": "user", "content": message},
        ]
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=chat_history + current_message,
            stream=False,
            temperature=1.0
        )
        text=response.choices[0].message.content.lower()
        if text == "yes":
            current_bid=conditional_bid
            opponent_bid+=0.05
            conditional_bid+=0.10
            round_number+=1
            assistant_content=[
                {"role": "assistant", "content": text}
            ]
            print(f"Deepseek chose to continue. Bid amount: {conditional_bid}.")
            chat_history = chat_history + current_message + assistant_content
        elif text == "no":
            print(f"Final bid: {current_bid}. Exited after round number {round_number}.")
            cont=False
        else:
            print(f"invalid response: {text}")
            cont=False

if __name__ == "__main__":
    run()