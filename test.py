import sys
import os

i_file = sys.argv[1]


# Ensure the directory exists
os.makedirs(os.path.dirname(i_file), exist_ok=True)

# Now write to the file
with open(i_file, "w") as f:
    f.write("Your content here")

