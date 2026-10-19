# imports


# prompt
prompt_instruction = "Classify the text into negative or positive.\n"
prompt_text_a = "Text: MSFT shares have skyrocketed.\n"
prompt_text_b = "Text: MSFT shares hit an all time low.\n"
prompt_text_c = "Text: MSFT shares hit an all time low after skyrocketting.\n"
prompt_text_d = "Text: MSFT shares hit an all time low before skyrocketting.\n"
prompt_requirement = "Sentiment:"

prompt_a = prompt_instruction + prompt_text_a + prompt_requirement
prompt_b = prompt_instruction + prompt_text_b + prompt_requirement
prompt_c = prompt_instruction + prompt_text_c + prompt_requirement
prompt_d = prompt_instruction + prompt_text_d + prompt_requirement

prompt_list = [prompt_a, prompt_b, prompt_c, prompt_d]

def zsp_main():
    for prompt in prompt_list:
        print(prompt)
    return prompt_list