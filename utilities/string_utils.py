def add_string_before_dot(input_string, string_to_add):
    if "." not in input_string:
        return input_string
    
    parts = input_string.split(".", 1)
    return f"{parts[0]}{string_to_add}.{parts[1]}"