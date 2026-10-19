# imports


# prompt
# Taken from stock_emotions_train.csv
prompt_sentiment_a = ""
prompt_sentiment_b = ""
prompt_sentiment_c = ""
prompt_sentiment_d = ""
prompt_sentiment = prompt_sentiment_a + prompt_sentiment_b + prompt_sentiment_c + prompt_sentiment_d

prompt_text_a = ""
prompt_text_b = ""
prompt_text_c = ""
prompt_text_d = ""

prompt_a = prompt_sentiment + prompt_text_a
prompt_b = prompt_sentiment + prompt_text_b
prompt_c = prompt_sentiment + prompt_text_c
prompt_d = prompt_sentiment + prompt_text_d

prompt_list = [prompt_a, prompt_b, prompt_c, prompt_d]

def sp_main():
    for prompt in prompt_list:
        print(prompt)
    return prompt_list