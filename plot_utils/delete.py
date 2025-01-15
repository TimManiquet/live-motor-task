import json

# Example dictionary
my_dict = {"key1": "value1", "key2": 2, "key3": [1, 2, 3]}

# Save dictionary to a file
with open('dictionary.json', 'w') as json_file:
    json.dump(my_dict, json_file, indent=4)  # `indent` makes the JSON human-readable

# To load it back:
with open('dictionary.json', 'r') as json_file:
    loaded_dict = json.load(json_file)

print(loaded_dict)