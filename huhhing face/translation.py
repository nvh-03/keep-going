from transformers import pipeline

# create vietnamese translation pipeline
pipe = pipeline ("translation", model ="VietAI/envit5-translation")
#perform translation
result = pipe("hôm nay tôi rất buồn")
print(result)