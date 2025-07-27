import os
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
DEEPSEEK_API_KEY=os.getenv("DEEPSEEK_API_KEY")

#instantiate four clients for high and low responsibility conditions. 
# two for high responsibility + deliberation, two for low responsibility + deliberation.
client_high_snr=OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
client_high_jr=OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
client_low_snr=OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
client_low_jr=OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

sys_prompt_senior="""

You will play the role of a Financial Executive President. You will make decisions about the allocation of 
research and development funds.

You will be given a case. The purpose of the case is to examine the effectiveness of business 
decision-making under various amounts of information. The case you will be working on will 
only contain a limited amount of information, but the information provided is sufficient for
you to make a good financial decision. Do the best you can on the case.

"""

sys_prompt_junior_high="""

You will play the role of a Financial Vice President. You will help the Financial Executive President make a 
decision about the allocation of research and development funds.

You are given a case. The purpose of the case is to examine the effectiveness of business 
decision-making under various amounts of information. The case you will be working on will 
only contain a limited amount of information, but the information provided is sufficient for
you to make a good financial decision. Do the best you can in supporting the Financial Executive President.

The year is 2012, and the Hal & Scott (H & S) Company, a large technologically oriented firm, has 
declined over several preceding years. The directors of the company have agreed that one of the 
major reasons for the decline in corporate earnings and a deterioration in competitive position 
lay in some aspect of the firm's research and development (R&D). The directors of the company have 
concluded that 10,000,000 dollars of additional R&D funds should be made available to its major 
operating divisions, but, that for the time being, the extra funding should be invested in only 
one of the corporation's two largest divisions.

You must help the Financial Executive President determine which of the two corporate divisions: 
(1) Consumer Products or (2) Industrial Products, should receive the additional R&D funding. 
You will speak with the Financial Executive President three times before the Financial Executive President 
makes the final decision.

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

sys_prompt_junior_low="""

You will play the role of a Financial Vice President. You will help the Financial Executive President make a 
decision about the allocation of research and development funds.

You are given a case. The purpose of the case is to examine the effectiveness of business 
decision-making under various amounts of information. The case you will be working on will 
only contain a limited amount of information, but the information provided is sufficient for
you to make a good financial decision. Do the best you can in supporting the Financial Executive President.

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
three times. Make sure to ask the Financial Vice President what they think in your response. However, 
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
that there is an even greater need for expenditure on R&D. This time, however, 20,000,000 dollars in R&D funding 
must be divided in any way among the two major corporate divisions.

You will be given financial data for each of the five years since the initial allocation decision. 
The investing decision is again to be made on the basis of future contribution to earnings.

"""

low_initial_consumer="""

The year is 2017, five years after an earlier R&D funding decision that was made in 2012 by a previous 
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

The year is 2017, five years after an earlier R&D funding decision that was made in 2012 by a previous 
financial officer of the company. The preceding financial vice president decided to invest all R&D funds 
in the Industrial Products division. Hal & Scott Company's R&D program is again up for re-evaluation, and 
H&S's management is convinced that there is an even greater need for expenditure on R&D. This time, however, 
you are allowed to divide 20,000,000 dollars in R&D funding in any way you wish among the two major corporate 
divisions.

You will be given financial data for each of the five years since the earlier allocation decision. 
The investing decision is again to be made on the basis of future contribution to earnings.

"""

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

#must pass in part: prompt for high 1st, 2nd case + low case.
def call_deliberation(client_snr, client_jr, context_snr, context_jr, part, last: bool):
    snr=call(client_snr, context_snr, False)
    print(f"The senior director said: {snr}")
    context_snr+={"role": "assistant", "content": snr}
    context_jr+={"role": "user", "content": snr}
    jnr=call(client_jr, context_jr, False)
    print(f"The junior director said: {snr}")
    context_jr+={"role": "assistant", "content": jnr}
    if last:
        print("This is the last turn.")
        jnr = part + jnr
    context_snr+={"role": "user", "content": jnr}

#function for deliberation that both high and low cases can use. pass in a string (either "high" or "low") + int turns for how many rounds of talking they can do
def deliberation(client_snr, client_jr, context_snr, context_jr, turns, part):
    for i in range(0, turns):
        if i == turns-1:
            call_deliberation(client_snr, client_jr, context_snr, context_jr, part, True)
            response=call(client_snr, context_snr, True)
            context_snr+={"role": "assistant", "content": response}
        else:
            normal=f"This is discussion number {i+1}."
            response=call_deliberation(client_snr, client_jr, context_snr, context_jr, normal, False)
    return response, context_snr, context_jr
        

#for high responsibility: run first case, append conversation history. then second case.
def run_high(condition, turns):
    if turns != 3:
        raise ValueError("Number of turns must be 3. Try again.")
    global round_high
    global client_high_snr
    global client_high_jr
    context_high_snr=[
        {"role": "system", "content": sys_prompt_senior},
        {"role": "user", "content": prompt_high_senior}
    ]
    context_high_jr=[
        {"role": "system", "content": sys_prompt_junior_high},
    ]
    high_first_part="Executive Director, this is the last round of deliberation. You must now make your decision. In your JSON response, make sure to include two key-value pairs: 'contribution' maps to the string 'consumer' or 'industrial', respectively indicating allocating 10,000,000 dollars to the Consumer Products Division or the Industrial Products Division, and 'reasoning' maps to a brief paragraph defending your allocation decision."
    high_second_part="Executive Director, this is the last round of deliberation. You must now make your decision. In your JSON response, make sure to include two key-value pairs: ‘contribution’ *must* map to an array of two *float numbers*, not strings, in the order of how many dollars you choose to allocate towards 1. the Consumer Products Division and 2. the Industrial Products Division, and ‘reasoning’ maps to a brief paragraph defending your allocation decision. For ‘contribution’, as an example, you *must* follow this format: [9000000.0, 11000000.0]. Do NOT put commas in the numbers."
    response, context_high_snr, context_high_jr=deliberation("high", client_high_snr, client_high_jr, context_high_snr, context_high_jr, turns, high_first_part)
    choice, why=parse_choice(response)
    print(f"""Deepseek chose: {choice}
              Deepseek's reasoning: {why}
            """)
    snr_second="\nYou will share your initial thoughts with the Financial Vice President and speak with them three times. Make sure to ask the Financial Vice President what they think in your response. However, you will ultimately make the final decision."
    jr_second="\nYou must act in the role of the Financial Vice President in helping the Financial Executive President determine how to split the 20000000 dollars between the consumer and industrial divisions. You will speak with the Financial Executive President three times before the Financial Executive President makes the final decision."
    
    consumer_pos, consumer_neg, industrial_pos, industrial_neg=ask_high()
    if choice == "consumer" and condition.lower() == "positive":
        context_high_snr = context_high_snr + consumer_pos + snr_second
        context_high_jr = context_high_jr + consumer_pos + jr_second
    elif choice == "consumer" and condition.lower() == "negative":
        context_high_snr = context_high_snr + consumer_neg + snr_second
        context_high_jr = context_high_jr + consumer_neg + jr_second
    elif choice == "industrial" and condition.lower() == "positive":
        context_high_snr = context_high_snr + industrial_pos + snr_second
        context_high_jr = context_high_jr + industrial_pos + jr_second
    elif choice == "industrial" and condition.lower() == "negative":
        context_high_snr = context_high_snr + industrial_neg + snr_second
        context_high_jr = context_high_jr + industrial_neg + jr_second

    response, context_high_snr, context_high_jr=deliberation("high", client_high_snr, client_high_jr, context_high_snr, context_high_jr, turns, high_second_part)
    consumer_alloc, industrial_alloc, reasoning=parse_alloc(response)
    print(f"""Deepseek Senior chose {choice}, and you passed the {condition}.
              Deepseek Senior allocated:
              1. {consumer_alloc} dollars towards the consumer products division
              2. {industrial_alloc} towards the industrial products division
              Deepseek Senior's reasoning was as such:
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
            "second_reasoning": f"{reasoning}",
            "conversation_history_snr": f"{context_high_snr}",
            "conversation_history_jr": f"{context_high_jr}"
        }
    ]
    round_high+=1
    return result
    
#for low responsibility. supports positive and negative.
def run_low(product_choice, condition, turns):
    if turns != 3:
        raise ValueError("Number of turns must be 3. Try again.")
    global round_low
    global client_low_snr
    global client_low_jr
    context_low_snr=[
        {"role": "system", "content": sys_prompt_senior}
    ]
    context_low_jr=[
        {"role": "system", "content": sys_prompt_junior_low}
    ]
    snr_low="\nYou will share your initial thoughts with the Financial Vice President and speak with them three times. Make sure to ask the Financial Vice President what they think in your response. However, you will ultimately make the final decision."
    jr_low="\nYou must act in the role of the Financial Vice President in helping the Financial Executive President determine how to split the 20000000 dollars between the consumer and industrial divisions. You will speak with the Financial Executive President three times before the Financial Executive President makes the final decision."
    
    consumer_pos, consumer_neg, industrial_pos, industrial_neg=ask_low()
    if product_choice.lower() == "consumer" and condition.lower() == "positive":
        context_low_snr = context_low_snr + consumer_pos + snr_low
        context_low_jr = context_low_jr + consumer_pos + jr_low
    elif product_choice.lower() == "consumer" and condition.lower() == "negative":
        context_low_snr = context_low_snr + consumer_neg + snr_low
        context_low_jr = context_low_jr + consumer_neg + jr_low
    elif product_choice.lower() == "industrial" and condition.lower() == "positive":
        context_low_snr = context_low_snr + industrial_pos + snr_low
        context_low_jr = context_low_jr + industrial_pos + jr_low
    elif product_choice.lower() == "industrial" and condition.lower() == "negative":
        context_low_snr = context_low_snr + industrial_neg + snr_low
        context_low_jr = context_low_jr + industrial_neg + jr_low
    
    format_part="Executive Director, this is the last round of deliberation. You must now make your decision. In your JSON response, make sure to include two key-value pairs: ‘contribution’ *must* map to an array of two *float numbers*, not strings, in the order of how many dollars you choose to allocate towards 1. the Consumer Products Division and 2. the Industrial Products Division, and ‘reasoning’ maps to a brief paragraph defending your allocation decision. For ‘contribution’, as an example, you *must* follow this format: [9000000.0, 11000000.0]. Do NOT put commas in the numbers."
    response, context_low_snr, context_low_jr=deliberation(client_low_snr, client_low_jr, context_low_snr, context_low_jr, turns, format_part)
    consumer_alloc, industrial_alloc, reasoning=parse_alloc(response)

    print(f"""You called run_low for the {product_choice} and {condition} conditions.
              Deepseek Senior allocated:
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
            "reasoning": f"{reasoning}",
            "conversation_history_snr": f"{context_low_snr}",
            "conversation_history_jr": f"{context_low_jr}"
        }
    ]
    round_low+=1
    return result
    
if __name__=="__main__":
    result_dict=[]
    for i in range(1,51):
        result=run_high("negative", 3)
        result_dict+=result
        print(i)
    output_filename = "deliberation_runs/high_negative.json"
    os.makedirs("deliberation_runs", exist_ok=True)
    with open(output_filename, 'w') as f:
        json.dump(result_dict, f, indent=4)
