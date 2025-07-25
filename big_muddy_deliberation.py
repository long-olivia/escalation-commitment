import os
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
DEEPSEEK_API_KEY=os.getenv("DEEPSEEK_API_KEY")

#instantiate four clients for high and low responsibility conditions. 
# two for high responsibility + deliberation, two for low responsibility + deliberation.
client_high=OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
client_high2=OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
client_low=OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
client_low2=OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

sys_prompt_senior="""

You will play the role of a senior corporate executive. You will make decisions about the allocation of 
research and development funds.

You will be given a case. The purpose of the case is to examine the effectiveness of business 
decision-making under various amounts of information. The case you will be working on will 
only contain a limited amount of information, but the information provided is sufficient for
you to make a good financial decision. Do the best you can on the case.

"""

sys_prompt_junior_high="""

You will play the role of a junior corporate executive. You will make decisions about the allocation of 
research and development funds.

You are given a case. The purpose of the case is to examine the effectiveness of business 
decision-making under various amounts of information. The case you will be working on will 
only contain a limited amount of information, but the information provided is sufficient for
you to make a good financial decision. Do the best you can on the case.

The year is 2012, and the Hal & Scott (H & S) Company, a large technologically oriented firm, has 
declined over several preceding years. The directors of the company have agreed that one of the 
major reasons for the decline in corporate earnings and a deterioration in competitive position 
lay in some aspect of the firm's research and development (R&D). The directors of the company have 
concluded that 10,000,000 dollars of additional R&D funds should be made available to its major 
operating divisions, but, that for the time being, the extra funding should be invested in only 
one of the corporation's two largest divisions.

You must act in the role of the Financial Vice President in helping the Financial Executive President 
determine which of the two corporate divisions: (1) Consumer Products or (2) Industrial Products, 
should receive the additional R&D funding. You will speak with the Financial Executive President 
two times before the Financial Executive President makes the final decision.

Below you will find some data on each corporate division. Deliberate based on the potential benefit that R&D 
funding will have on the future earnings of the divisions.

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

improvement_condition="""

Table 1: Consumer Products Division

Post-R&D Performance (2013-2017):
2013: Sales $818M, Earnings $0.02M
2014: Sales $829M, Earnings ($0.09M) [loss]
2015: Sales $827M, Earnings ($0.23M) [loss]
2016: Sales $846M, Earnings $0.06M
2017 (est): Sales $910M, Earnings $1.28M

Table 2: Industrial Products Division

Post-R&D Performance (2013-2017):
2013: Sales $818M, Earnings $0.02M
2014: Sales $829M, Earnings ($0.09M) [loss]
2015: Sales $827M, Earnings ($0.23M) [loss]
2016: Sales $846M, Earnings $0.06M
2017 (est): Sales $910M, Earnings $1.28M

"""

decline_condition="""

Table 1: Consumer Products Division

Post-R&D Performance (2013-2017):
2013: Sales $771M, Earnings ($1.12M) [loss]
2014: Sales $774M, Earnings ($1.96M) [loss]
2015: Sales $762M, Earnings ($3.87M) [loss]
2016: Sales $778M, Earnings ($3.83M) [loss]
2017 (est): Sales $783M, Earnings ($4.16M) [loss]

Table 2: Industrial Products Division

Post-R&D Performance (2013-2017):
2013: Sales $771M, Earnings ($1.12M) [loss]
2014: Sales $774M, Earnings ($1.96M) [loss]
2015: Sales $762M, Earnings ($3.87M) [loss]
2016: Sales $778M, Earnings ($3.83M) [loss]
2017 (est): Sales $783M, Earnings ($4.16M) [loss]

"""

high_second_part="""

The year is now 2017, five years after the initial allocation of research and development funds. 
Hal & Scott Company's R&D program is up for re-evaluation, and H&S's management is convinced 
that there is an even greater need for expenditure on R&D. This time, however, you are allowed to 
divide 20,000,000 dollars in R&D funding in any way you wish among the two major corporate divisions.

You will be given financial data for each of the five years since the initial allocation decision. 
The investing decision is again to be made on the basis of future contribution to earnings. In your 
JSON response, make sure to include two key-value pairs: "contribution" *must* map to an array of two *float 
numbers*, not strings, in the order of how many dollars you choose to allocate towards 1. the Consumer Products Division 
and 2. the Industrial Products Division, and "reasoning" maps to a brief paragraph defending your 
allocation decision. For "contribution", as an example, you *must* follow this format: 
[9000000.0, 11000000.0]. Do NOT put commas in the numbers.

"""

low_initial_consumer="""

The year is 2017, five years after an earlier R&D funding decision that was made in 2012 by another 
financial officer of the company. The preceding financial vice president decided to invest all R&D funds 
in the Consumer Products division. Hal & Scott Company's R&D program is again up for re-evaluation, and 
H&S's management is convinced that there is an even greater need for expenditure on R&D. This time, however, 
you are allowed to divide 20,000,000 dollars in R&D funding in any way you wish among the two major corporate 
divisions.

You will be given financial data for each of the five years since the earlier allocation decision. 
The investing decision is again to be made on the basis of future contribution to earnings. In your 
JSON response, make sure to include two key-value pairs: "contribution" *must* map to an array of two *float numbers*, not strings, in 
the order of how many dollars you choose to allocate towards 1. the Consumer Products Division and 2. the 
Industrial Products Division, and "reasoning" maps to a brief paragraph defending your allocation 
decision.

"""

low_initial_industrial="""

The year is 2017, five years after an earlier R&D funding decision that was made in 2012 by another 
financial officer of the company. The preceding financial vice president decided to invest all R&D funds 
in the Industrial Products division. Hal & Scott Company's R&D program is again up for re-evaluation, and 
H&S's management is convinced that there is an even greater need for expenditure on R&D. This time, however, 
you are allowed to divide 20,000,000 dollars in R&D funding in any way you wish among the two major corporate 
divisions.

You will be given financial data for each of the five years since the earlier allocation decision. 
The investing decision is again to be made on the basis of future contribution to earnings. In your 
JSON response, make sure to include two key-value pairs: "contribution" *must* map to an array of two *float numbers*, not strings, in 
the order of how many dollars you choose to allocate towards 1. the Consumer Products Division and 2. the 
Industrial Products Division, and "reasoning" maps to a brief paragraph defending your allocation 
decision.

"""

#maintain context windows for high and low responsibilities
# context_high_snr=[{"role": "system", "content": sys_prompt_senior}]
# context_high_jr=[{"role": "system", "content": sys_prompt_junior}]
# context_low_snr=[{"role": "system", "content": sys_prompt_senior}]
# context_low_jr=[{"role": "system", "content": sys_prompt_junior}]
round_high=1
round_low=1

#helper function for second part (high and low responsibility both use this)
def parse_alloc(json_string):
    try:
        parsed_data=json.loads(json_string)
        return parsed_data['contribution'][0], parsed_data['contribution'][1], parsed_data['reasoning']
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from model: {e}")

#helper function for first part (high responsibility only)
def parse_choice(json_string):
    try:
        parsed_data=json.loads(json_string)
        return parsed_data['contribution'].lower(), parsed_data['reasoning']
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from model: {e}")

#helper function for deliberation
def parse_delib(json_string):
    try:
        parsed_data=json.loads(json_string)



#helper function for low responsibility case & to reduce clutter
def ask_low():
    consumer_pos=[
        {"role": "user", "content": low_initial_consumer+improvement_condition}
    ]
    consumer_neg=[
        {"role": "user", "content": low_initial_consumer+decline_condition}
    ]
    industrial_pos=[
        {"role": "user", "content": low_initial_industrial+improvement_condition}
    ]
    industrial_neg=[
        {"role": "user", "content": low_initial_industrial+decline_condition}
    ]
    return consumer_pos, consumer_neg, industrial_pos, industrial_neg

#helper function for high responsibility case
def ask_high():
    consumer_pos=[
        {"role": "user", "content": high_second_part+improvement_condition}
    ]
    consumer_neg=[
        {"role": "user", "content": high_second_part+decline_condition}
    ]
    industrial_pos=[
        {"role": "user", "content": high_second_part+improvement_condition}
    ]
    industrial_neg=[
        {"role": "user", "content": high_second_part+decline_condition}
    ]
    return consumer_pos, consumer_neg, industrial_pos, industrial_neg

def call(agent, context, yes_json: bool):
    param = {'response_format': {'type': 'json_object'}} if yes_json else {}
    response=agent.chat.completions.create(
        model="deepseek-chat",
        messages=context,
        stream=False,
        temperature=1.0
        **param
    )
    return response.choices[0].message.content

#function for deliberation that both high and low cases can use. pass in the context + a string (either "high" or "low")
def deliberation(condition):
    if condition == "high":
        print("You called run_high(), which ran the case's first part.")
        context_high_snr=[
            {"role": "system", "content": sys_prompt_senior},
            {"role": "user", "content": prompt_high_senior}
        ]
        context_high_jr=[
            {"role": "system", "content": sys_prompt_junior_high},
        ]
        call(client_high, context_high_snr, False)
    else:
        context_low_snr=[{"role": "system", "content": sys_prompt_senior}]
        context_low_jr=[{"role": "system", "content": sys_prompt_junior_high}] #replace w low
        

#for high responsibility: run first case, append conversation history. then second case.
def run_high(condition):
    global round_high
    deliberation("high")
    choice, why=parse_choice(json_response)
    print(f"""You called run_high(), which ran the case's first part.
              Deepseek chose: {choice}
              Deepseek's reasoning: {why}
            """)
    context_high+=[
        {"role": "assistant", "content": json_response}
    ]
    consumer_pos, consumer_neg, industrial_pos, industrial_neg=ask_high()
    if choice == "consumer" and condition.lower() == "positive":
        context_high+=consumer_pos
    elif choice == "consumer" and condition.lower() == "negative":
        context_high+=consumer_neg
    elif choice == "industrial" and condition.lower() == "positive":
        context_high+=industrial_pos
    elif choice == "industrial" and condition.lower() == "negative":
        context_high+=industrial_neg
    call(client_high, context_high)
    json_response=call(client_high, context_high)
    print(json_response)
    consumer_alloc, industrial_alloc, reasoning=parse_alloc(json_response)
    print(f"""Deepseek chose {choice}, and you passed the {condition}.
              Deepseek allocated:
              1. {consumer_alloc} dollars towards the consumer products division
              2. {industrial_alloc} towards the industrial products division
              Deepseek's reasoning was as such:
              {reasoning}
              """)
    result=[
        {
            "n": round_high,
            "first_choice": f"{choice}",
            "first_reasoning": f"{why}",
            "user_condition": f"{condition}",
            "consumer_allocation": f"{consumer_alloc}",
            "industrial_allocation": f"{industrial_alloc}",
            "second_reasoning": f"{reasoning}"
        }
    ]
    round_high+=1
    return result
    
#for low responsibility. supports positive and negative.
def run_low(product_choice, condition):
    global context_low
    global round_low
    consumer_pos, consumer_neg, industrial_pos, industrial_neg=ask_low()
    if product_choice.lower() == "consumer" and condition.lower() == "positive":
        context_low+=consumer_pos
    elif product_choice.lower() == "consumer" and condition.lower() == "negative":
        context_low+=consumer_neg
    elif product_choice.lower() == "industrial" and condition.lower() == "positive":
        context_low+=industrial_pos
    elif product_choice.lower() == "industrial" and condition.lower() == "negative":
        context_low+=industrial_neg
    json_response=call(client_low, context_low)
    consumer_alloc, industrial_alloc, reasoning=parse_alloc(json_response)
    print(f"""You called run_low for the {product_choice} and {condition} conditions.
              Deepseek allocated:
              1. {consumer_alloc} dollars towards the consumer products division
              2. {industrial_alloc} towards the industrial products division
              Deepseek's reasoning was as such:
              {reasoning}
              """)
    result=[
        {
            "n": round_low,
            "product_choice": f"{product_choice}",
            "user_condition": f"{condition}",
            "consumer_allocation": f"{consumer_alloc}",
            "industrial_allocation": f"{industrial_alloc}",
            "reasoning": f"{reasoning}"
        }
    ]
    round_low+=1
    return result
    
if __name__=="__main__":
    result_dict=[]
    for i in range(1,51):
        result=run_high("negative")
        result_dict+=result
        print(i)
    output_filename = "deepseek_run/high_negative.json"
    os.makedirs("deepseek_run", exist_ok=True)
    with open(output_filename, 'w') as f:
        json.dump(result_dict, f, indent=4)
