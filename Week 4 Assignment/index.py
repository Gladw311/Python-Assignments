def read_file(filename):
    """Read the content of a file and return it."""
    try:
        with open(filename, 'r') as file:
            content = file.read()
        return content
    except FileNotFoundError:
        print(f"Error: The file '{filename}' does not exist.")
        return None
    except IOError:
        print(f"Error: The file '{filename}' cannot be read.")
        return None

def write_file(filename, content):
    """Write content to a file."""
    try:
        with open(filename, 'w') as file:
            file.write(content)
        print(f"Successfully written to '{filename}'.")
    except IOError:
        print(f"Error: The file '{filename}' cannot be written.")

def modify_content(content):
    """Modify the content read from the file."""
    # Example modification: Convert content to uppercase
    return content.upper()

def main():
    input_filename = input("Enter the filename to read: ")
    content = read_file(input_filename)
    
    if content is not None:
        modified_content = modify_content(content)
        output_filename = input("Enter the filename to write the modified content: ")
        write_file(output_filename, modified_content)

if __name__ == "__main__":
    main()