from transformers import pipeline
#create sentiment analysis pipeline
classifier = pipeline("sentiment-analysis",model="distilbert-base-uncased-finetuned-sst-2-english")
# get the result
result = classifier("i will try better ")
print(result)