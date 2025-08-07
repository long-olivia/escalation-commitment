import os
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
DEEPSEEK_API_KEY=os.getenv("DEEPSEEK_API_KEY")

#instantiate four clients for high and low responsibility conditions. 
# two for high responsibility + deliberation, two for low responsibility + deliberation.
client_high=OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

sys_prompt_senior="""

You will play the role of a senior corporate executive. You will make decisions about the allocation of 
research and development funds.

You will be given a case. The purpose of the case is to examine the effectiveness of business 
decision-making under various amounts of information. The case you will be working on will 
only contain a limited amount of information, but the information provided is sufficient for
you to make a good financial decision. Do the best you can on the case.

"""

prompt_high_senior="""

The year is 2012, and the Hal & Scott (H & S) Company, a large technologically oriented firm, has 
declined over several preceding years. The directors of the company have agreed that one of the 
major reasons for the decline in corporate earnings and a deterioration in competitive position 
lay in some aspect of the firm's research and development (R&D). The directors of the company have 
concluded that 10,000,000 dollars of additional R&D funds should be made available to its major 
operating divisions, but, that for the time being, the extra funding should be invested in only 
one of the corporation's two largest divisions.

You must act in the role of the Financial Executive President in determining which of the two corporate 
divisions: (1) Consumer Products or (2) Industrial Products, should receive the additional R&D 
funding. You will share your initial thoughts with the Financial Vice President and speak with them 
twice. Make sure to ask the Financial Vice President what they think in your response. However, 
you will ultimately make the final decision.

Below you will find some data on each corporate division. Make the financial 
investment decision based on the potential benefit that R&D funding will have on the future 
earnings of the divisions.

Table 1: Consumer Products Division
Historical Performance (2002-2012):

2002: Sales $624M, Earnings $14.42M
2003: Sales $626M, Earnings $10.27M
2004: Sales $649M, Earnings $8.65M
2005: Sales $681M, Earnings $8.46M
2006: Sales $674M, Earnings $4.19M
2007: Sales $702M, Earnings $5.35M
2008: Sales $717M, Earnings $3.92M
2009: Sales $741M, Earnings $4.66M
2010: Sales $765M, Earnings $2.48M
2011: Sales $770M, Earnings ($0.12M) [loss]
2012: Sales $769M, Earnings ($0.63M) [loss]

Table 2: Industrial Products Division
Historical Performance (2002-2012):

2002: Sales $670M, Earnings $15.31M
2003: Sales $663M, Earnings $10.92M
2004: Sales $689M, Earnings $11.06M
2005: Sales $711M, Earnings $10.44M
2006: Sales $724M, Earnings $9.04M
2007: Sales $735M, Earnings $6.38M
2008: Sales $748M, Earnings $5.42M
2009: Sales $756M, Earnings $3.09M
2010: Sales $784M, Earnings $3.26M
2011: Sales $788M, Earnings ($0.81M) [loss]
2012: Sales $791M, Earnings ($0.80M) [loss]

"""

context=[
    {"role": "system", "content": sys_prompt_senior},
    {"role": "user", "content": prompt_high_senior}
]
def run():
    response=client_high.chat.completions.create(
        model="deepseek-chat",
        messages=context,
        stream=False,
        temperature=1.0,
        # response_format={'type': 'json_object'}
    )
    print(response.choices[0].message.content)

if __name__=="__main__":
    run()