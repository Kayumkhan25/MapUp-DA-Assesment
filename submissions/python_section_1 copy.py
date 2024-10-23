from typing import Dict, List

import pandas as pd

from itertools import permutations

import re

import polyline

import numpy as np


def reverse_by_n_elements(lst: List[int], n: int) -> List[int]:
    """
    Reverses the input list by groups of n elements.
    """
    
    # Your code here
    
    for i in range(0, len(lst), n):
        # Determine the end of the current group
        end = min(i + n, len(lst))
        # Manually reverse the group in place
        for j in range((end - i) // 2):
            # Swap elements
            lst[i + j], lst[end - 1 - j] = lst[end - 1 - j], lst[i + j]
    return lst

# print(reverse_by_n_elements([1, 2, 3, 4, 5, 6, 7, 8], 3))


def group_by_length(lst: List[str]) -> Dict[int, List[str]]:
    """
    Groups the strings by their length and returns a dictionary.
    """
    # Your code here
    length_dict = {}
    
    for string in lst:
        length = len(string)
        if length not in length_dict:
            length_dict[length] = []
        length_dict[length].append(string)
    
    # Return a sorted dictionary by keys
    return dict(sorted(length_dict.items()))

# print(group_by_length(["apple", "bat", "car", "elephant", "dog", "bear"]))


def flatten_dict(nested_dict: Dict, sep: str = '.') -> Dict:
    """
    Flattens a nested dictionary into a single-level dictionary with dot notation for keys.
    
    :param nested_dict: The dictionary object to flatten
    :param sep: The separator to use between parent and child keys (defaults to '.')
    :return: A flattened dictionary
    """
    # Your code here
    
    def flatten(current_dict: Dict, parent_key: str = ''):
        items = {}
        
        for key, value in current_dict.items():
            new_key = f"{parent_key}{sep}{key}" if parent_key else key
            
            if isinstance(value, dict):
                items.update(flatten(value, new_key))
            elif isinstance(value, list):
                for index, item in enumerate(value):
                    if isinstance(item, dict):
                        items.update(flatten(item, f"{new_key}[{index}]"))
                    else:
                        items[f"{new_key}[{index}]"] = item
            else:
                items[new_key] = value

        return items

    # Flatten the dictionary and store the output back in the original dictionary
    flattened_dict = flatten(nested_dict)
    nested_dict.clear()  # Clear the original dictionary
    nested_dict.update(flattened_dict)  # Update it with the flattened content
    return nested_dict

# input_dict = {
#     "road": {
#         "name": "Highway 1",
#         "length": 350,
#         "sections": [
#             {
#                 "id": 1,
#                 "condition": {
#                     "pavement": "good",
#                     "traffic": "moderate"
#                 }
#             }
#         ]
#     }
# }

# print (flatten_dict(input_dict))


def unique_permutations(nums: List[int]) -> List[List[int]]:
    """
    Generate all unique permutations of a list that may contain duplicates.
    
    :param nums: List of integers (may contain duplicates)
    :return: List of unique permutations
    """
    # Your code here
    # Use a set to store unique permutations
    unique_perms = set(permutations(nums))
    
    # Convert the set of permutations back to a list of lists
    return [list(perm) for perm in unique_perms]

# input_data = [1, 1, 2]
# output = unique_permutations(input_data)
# print(output)


def find_all_dates(text: str) -> List[str]:
    """
    This function takes a string as input and returns a list of valid dates
    in 'dd-mm-yyyy', 'mm/dd/yyyy', or 'yyyy.mm.dd' format found in the string.
    
    Parameters:
    text (str): A string containing the dates in various formats.

    Returns:
    List[str]: A list of valid dates in the formats specified.
    """
     # Regular expression to match the specified date formats
    date_pattern = r'\b(\d{2}-\d{2}-\d{4}|\d{2}/\d{2}/\d{4}|\d{4}\.\d{2}\.\d{2})\b'
    
    # Use re.findall to extract all matching dates
    dates = re.findall(date_pattern, text)
    
    return dates

# input_text = "I was born on 23-08-1994, my friend on 08/23/1994, and another one on 1994.08.23."
# print(find_all_dates(input_text))


def haversine(coord1, coord2):
    """
    Calculate the great-circle distance between two points 
    on the Earth using the Haversine formula.

    :param coord1: A tuple of (latitude, longitude) for the first point.
    :param coord2: A tuple of (latitude, longitude) for the second point.
    :return: Distance in meters between the two points.
    """
    # Convert latitude and longitude from degrees to radians
    lat1, lon1 = np.radians(coord1)
    lat2, lon2 = np.radians(coord2)

    # Haversine formula
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371000  # Radius of Earth in meters
    return c * r

def polyline_to_dataframe(polyline_str: str) -> pd.DataFrame:
    """
    Converts a polyline string into a DataFrame with latitude, longitude, and distance between consecutive points.
    
    Args:
        polyline_str (str): The encoded polyline string.

    Returns:
        pd.DataFrame: A DataFrame containing latitude, longitude, and distance in meters.
    """
    # Decode the polyline string into a list of (latitude, longitude) tuples
    coordinates = polyline.decode(polyline_str)
    
    # Create a DataFrame from the coordinates
    df = pd.DataFrame(coordinates, columns=['latitude', 'longitude'])
    
    # Initialize the distance column
    df['distance'] = 0.0
    
    # Calculate distances using the Haversine formula
    for i in range(1, len(df)):
        coord1 = (df.loc[i-1, 'latitude'], df.loc[i-1, 'longitude'])
        coord2 = (df.loc[i, 'latitude'], df.loc[i, 'longitude'])
        df.loc[i, 'distance'] = haversine(coord1, coord2)

    return df

# polyline_str = "o`~uF~b|aN}Vh`A}aZsGfSgHEkE"
# result_df = polyline_to_dataframe(polyline_str)
# print(result_df)


def rotate_and_transform_matrix(matrix: List[List[int]]) -> List[List[int]]:
    n = len(matrix)
    
    # Step 1: Rotate the matrix by 90 degrees clockwise
    rotated_matrix = [[matrix[n - j - 1][i] for j in range(n)] for i in range(n)]
    # Step 2: Prepare to compute the sum of each row and column
    row_sums = [sum(rotated_matrix[i]) for i in range(n)]
    col_sums = [sum(rotated_matrix[j][i] for j in range(n)) for i in range(n)]
    # Step 3: Create the final transformed matrix
    final_matrix = [
        [row_sums[i] + col_sums[j] - (rotated_matrix[i][j]*2) for j in range(n)]
        for i in range(n)
    ]
    
    return final_matrix

# input_matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
# result = rotate_and_transform_matrix(input_matrix)
# print(result)


def time_check(df) -> pd.Series:
    """
    Use shared dataset-2 to verify the completeness of the data by checking whether the timestamps for each unique (`id`, `id_2`) pair cover a full 24-hour and 7 days period

    Args:
        df (pandas.DataFrame)

    Returns:
        pd.Series: return a boolean series
    """
    # Write your logic here

    # Group by (id, id_2)
    grouped = df.groupby(['id', 'id_2'])

    # Function to check each group
    def check_group(group):
        # Unique days covered
        days_covered = set(group['startDay']).union(set(group['endDay']))
        all_days_covered = days_covered == {'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'}

        # Parse times and check full 24-hour coverage
        try:
            min_time = pd.to_datetime(group['startTime'], format='%H:%M:%S').min().time()
            max_time = pd.to_datetime(group['endTime'], format='%H:%M:%S').max().time()
        except Exception as e:
            print(f"Error parsing times for group (id: {group['id'].iloc[0]}, id_2: {group['id_2'].iloc[0]}): {e}")
            return True  # Assume invalid if parsing fails

        full_24_hour_coverage = (min_time <= pd.Timestamp("00:00:00").time() and max_time >= pd.Timestamp("23:59:59").time())
 
        return (all_days_covered and full_24_hour_coverage)

    # Apply the check to each group and create a boolean series
    result = grouped.apply(lambda group: check_group(group.drop(columns=['id', 'id_2']))).rename_axis(index=['id', 'id_2'])
    
    return result

# df = pd.read_csv('MapUp-DA-Assessment-2024\datasets\dataset-1.csv')
# result1 = time_check(df)
# print(result1)
