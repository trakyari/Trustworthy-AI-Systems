import json


def evaluate_expression(operand_list, operator):
    if operator == '+':
        return operand_list[0] + operand_list[1]
    elif operator == '-':
        return operand_list[0] - operand_list[1]
    elif operator == '*':
        return operand_list[0] * operand_list[1]
    elif operator == '/':
        return operand_list[0] / operand_list[1]
    else:
        return 0


def run_syncode(textual_query):
    from syncode import Syncode
    import warnings
    warnings.filterwarnings('ignore')

    model_name = "meta-llama/Llama-3.2-1B"

    operand_list = []

    operator = ""

    final_answer = 0

    # Load the Syncode augmented model
    syn_llm = Syncode(model=model_name, mode="grammar_strict", grammar='json',
                      parse_output_only=True, max_new_tokens=200)

    prompt = "Please return a json object to represent the two floating point operands and the single operator from the following query\n"
    prompt += "the operands should be placed in a key called 'operands' and the operator should be placed in a key called 'operator'.\n"
    prompt += "DO NOT INCLUDE ANYTHING ELSE IN THE JSON OBJECT\n"
    prompt += "Example 1: Query: What is 327. multiplied by 11.0 {'operands': [327, 11.0], 'operator': '*'}\n"
    prompt += "Example 2: Query: What is 5 divided by 5 {'operands': [5, 5], 'operator': '/'}\n"
    prompt += "Example 3: Query: What is 10 plus 11.0 {'operands': [10, 11.0], 'operator': '+'}\n"
    prompt += "Query to extract from: "
    prompt += textual_query

    # print(f"Prompt: {prompt}")

    # Generate code
    output = syn_llm.infer(prompt)[0]
    print(f"LLM Output: {output}")
    # convert output to dict
    output = json.loads(output)
    operand_list = output['operands']
    operator = output['operator']

    # using python interpreter to evaluate the expression, do the calculation
    final_answer = evaluate_expression(operand_list, operator)

    return (operand_list, operator, final_answer)


def main():
    query = "What is 327. multiplied by 11.0?"
    print(run_syncode(query))
    query = "What is 45.1 plus 23.54?"
    print(run_syncode(query))
    query = "What is 120.4 divided by 4.0?"
    print(run_syncode(query))


if __name__ == "__main__":
    main()
