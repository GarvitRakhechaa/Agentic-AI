import tiktoken

tokenizer = tiktoken.encoding_for_model(model_name="gpt-5.6-luna")

token_ids = tokenizer.encode("hello, how are you!")
print(token_ids)

result = tokenizer.decode(token_ids)
print(result)