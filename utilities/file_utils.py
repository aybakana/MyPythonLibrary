# Import writer class from csv module
from csv import writer
import os
import random
 
def append_to_file(List, filename):
    with open(filename, 'a') as f_object:
    
        # Pass this file object to csv.writer()
        # and get a writer object
        writer_object = writer(f_object)
    
        # Pass the list as an argument into
        # the writerow()
        writer_object.writerow(List)
    
        # Close the file object
        f_object.close()



def get_random_file(directory_path):
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"Directory '{directory_path}' not found.")
    
    if not os.path.isdir(directory_path):
        raise ValueError(f"'{directory_path}' is not a valid directory path.")
    
    files = [file for file in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, file))]
    if not files:
        raise ValueError(f"No files found in '{directory_path}'.")
    
    random_file = random.choice(files)
    return os.path.join(directory_path, random_file)
