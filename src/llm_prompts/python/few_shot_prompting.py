# imports


# prompt
# Taken from stock_emotions_train.csv
prompt_sentiment_a = "$AMZN Dow futures up by 100 points already // Positive.\n"
prompt_sentiment_b = "$AAPL where are all the peeps posting bearish since $200 // Negative.\n"
prompt_sentiment_c = "$TSLA making money 💰 love to short this pig 🐷 // Negative.\n"
prompt_sentiment_d = "$TSLA 400s day is today🤩 // Positive.\n"
prompt_sentiment = prompt_sentiment_a + prompt_sentiment_b + prompt_sentiment_c + prompt_sentiment_d

prompt_text_a = "Text: MSFT shares have skyrocketed.\n"
prompt_text_b = "Text: MSFT shares hit an all time low.\n"
prompt_text_c = "Text: MSFT shares hit an all time low after skyrocketting.\n"
prompt_text_d = "Text: MSFT shares hit an all time low before skyrocketting.\n"

prompt_a = prompt_sentiment + prompt_text_a
prompt_b = prompt_sentiment + prompt_text_b
prompt_c = prompt_sentiment + prompt_text_c
prompt_d = prompt_sentiment + prompt_text_d

prompt_list = [prompt_a, prompt_b, prompt_c, prompt_d]

def fsp_main():
    for prompt in prompt_list:
        print(prompt)
    return prompt_list