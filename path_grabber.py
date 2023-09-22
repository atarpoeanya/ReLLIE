import os

def write_relative_paths_to_txt(folder_path, output_file):
    try:
        # Get a list of all files in the specified folder
        files = [os.path.join(root, filename) for root, _, filenames in os.walk(folder_path) for filename in filenames]

        # Create or open the output text file in write mode
        with open(output_file, 'w') as f:
            # Write each relative file path to the text file
            for file in files:
                relative_path = os.path.relpath(file, folder_path)
                f.write(relative_path + '\n')

        print(f"Relative paths written to '{output_file}' successfully.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

# Example usage:
folder_path = './data/plasmo/dataset'
output_file = './data/plasmo/low.txt'
write_relative_paths_to_txt(folder_path, output_file)
