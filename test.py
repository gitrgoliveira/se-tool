import ollama

print(ollama.list())
print("------------------")
print(ollama.show('solar:10.7b-instruct-v1-fp16')['template'])
# Output:
# ### System:
# {{ .System }}

# ### User:
# {{ .Prompt }}

# ### Assistant:

# Convert to python f-string:
print("------------------")
prompt = ollama.show('solar:10.7b-instruct-v1-fp16')['template'].replace("{{ .System }}", "{system}").replace("{{ .Prompt }}", "{prompt}")
print(prompt)
print(prompt.format(system="You are an assistant that proposes an alternative way of writing. ", prompt="Use the input text to {instruction}"))
