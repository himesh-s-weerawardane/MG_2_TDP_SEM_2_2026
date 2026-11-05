# Context
I want to evaluate the sentiment of a piece of social media financial commentary and label it as either "Positive" or "Negative".

# Objective
Classify the financial commentary as "Positive" (bullish: the writer expects the stock or market to go up, or is pleased about it) or "Negative" (bearish: the writer expects it to go down, or is unhappy about it).

# Examples
Text: "$AMZN Dow futures up by 100 points already." // Positive
Text: "$AAPL where are all the peeps posting bearish since $200" // Negative
Text: "$TSLA making money 💰 love to short this pig 🐷" // Negative
Text: "$TSLA 400s day is today🤩." // Positive

# Output format
Respond with a JSON object with two fields: "label", either "Positive" or "Negative", and "reason", one short sentence of at most 15 words saying what in the text decided the label. No other text.
