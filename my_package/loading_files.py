
import pandas as pd
import os
import io
from IPython.display import display
import ipywidgets as widgets
import numpy as np
import re


def excel_load_by_date(folder_path):
    ''' Loads all files from the specified folder, extracts the date from cell C2 of the second sheet (Sheet 1) of each file,
    and concatenates them into a dataframe according to the dates extracted'''
    # List all files in the folder
    files = os.listdir(folder_path)

    # Filter and sort the files alphabetically
    excel_files = [f for f in files if f.endswith('.xlsx') or f.endswith('.xls')]
    
    # Initialize an empty list to store dataframes with their corresponding dates
    df_list = []
    file_dates = []

    # Read cell C2 from the second sheet of each Excel file, extract the date, and store the dataframe with the date
    for file in excel_files:
        file_path = os.path.join(folder_path, file)
        df = pd.read_excel(file_path, sheet_name=1)  # Sheet 1 is the second sheet (0-indexed)
        date_value = df.iloc[1, 2]  # C2 is the second row (1-indexed) and third column (0-indexed)
        df['Date'] = date_value  # Add a new column 'Date' with the extracted date
        df_list.append(df)
        file_dates.append((file, date_value))

    # Sort files based on the extracted dates
    sorted_files = sorted(file_dates, key=lambda x: x[1])
    
    # Print the ordered dates
    print("Ordered dates:", [date for _, date in sorted_files])

    # Concatenate all dataframes into a single dataframe according to the sorted order
    sorted_df_list = [df_list[excel_files.index(file)] for file, _ in sorted_files]
    combined_df = pd.concat(sorted_df_list, ignore_index=True)
    
    return combined_df



def excel_to_df_from_folder(folder_path, sheet_number):
    ''' Loads all files from the specified folder in alphabetical 
    order and concatenates them into a dataframe'''
    # List all files in the folder
    files = os.listdir(folder_path)

    # Sort the files alphabetically
    excel_files = sorted(
        [f for f in files if f.endswith('.xlsx') or f.endswith('.xls')]
    )
    
    # Print the filenames in the order they will be processed
    print("Files in order:", excel_files)
   
    # Read each specified sheet from the Excel files and concatenate them into a single DataFrame
    df_list = [pd.read_excel(os.path.join(folder_path, file), sheet_name=sheet_number) for file in excel_files]
    combined_df = pd.concat(df_list, ignore_index=True)
    
    return combined_df



def excel_read_date(folder_path):
    ''' Loads all files from the specified folder in alphabetical 
    order and prints the content of cell C2 from the second sheet (Sheet 1) of each file'''
    # List all files in the folder
    files = os.listdir(folder_path)

    # Sort the files alphabetically
    excel_files = sorted(
        [f for f in files if f.endswith('.xlsx') or f.endswith('.xls')]
    )
    
    # Print the filenames in the order they will be processed
    print("Files in order:", excel_files)
   
    # Read cell C2 from the second sheet of each Excel file and print the content
    for file in excel_files:
        file_path = os.path.join(folder_path, file)
        df = pd.read_excel(file_path, sheet_name=1)  # Sheet 1 is the second sheet (0-indexed)
        cell_value = df.iloc[1, 2]  # C2 is the second row (1-indexed) and third column (0-indexed)
        print(f"Content of cell C2 in {file}: {cell_value}")


def csv_to_df_from_folder(folder_path):
    # List all CSV files in the folder
    files = [f for f in os.listdir(folder_path) if f.endswith('.CSV')]

    # Function to extract the number after the last underscore before ".CSV"
    def extract_number(filename):
        match = re.search(r'_(\d+)\.CSV$', filename, re.IGNORECASE)
        return int(match.group(1)) if match else float('inf')

    # Sort the files using natural order
    csv_files = sorted(files, key=extract_number)

    # Print the sorted file list for debugging
    print("Sorted files:", csv_files)

    # Read each CSV file and concatenate them into a single DataFrame
    df_list = [pd.read_csv(os.path.join(folder_path, file)) for file in csv_files]
    combined_df = pd.concat(df_list, ignore_index=True)

    return combined_df




def csvs_to_df(callback):
    ''' This function opens a file upload widget allowing the selection 
    of one or several csv files and joining them into a dataframe 
    in the order specified by the number at the end of their filenames.
    '''
    # Create a file upload widget
    upload_widget = widgets.FileUpload(
        accept='.CSV',  # Accept CSV files only
        multiple=True  # Accept multiple files
    )

    # Function to handle file upload and load CSVs into a combined DataFrame
    def on_file_upload(change):
        # Get the list of filenames and sort them by numeric order before loading data
        files = sorted(upload_widget.value.items(), 
                       key=lambda item: int(''.join(filter(str.isdigit, item[0]))))
        
        # Initialize an empty DataFrame to store the combined data
        df_raw = pd.DataFrame()
        
        # Load each file one by one and append to the main DataFrame
        for filename, file_info in files:
            df = pd.read_csv(io.BytesIO(file_info['content']))
            df['filename'] = filename  # Add a column with the filename
            df_raw = pd.concat([df_raw, df], ignore_index=True)
        
        # Display the combined DataFrame
        display(df_raw)
        
        # Call the callback function with the combined DataFrame
        callback(df_raw)

    # Attach the handler to the file upload widget
    upload_widget.observe(on_file_upload, names='value')

    # Display the file upload widget
    display(upload_widget)