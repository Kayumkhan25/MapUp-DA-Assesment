import pandas as pd

import numpy as np

from datetime import time


def calculate_distance_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate a distance matrix based on the dataframe, df.

    Args:
        df (pandas.DataFrame): A DataFrame with columns 'From_ID', 'To_ID', and 'Distance'.

    Returns:
        pandas.DataFrame: Distance matrix with cumulative distances.
    """
    # Create a unique list of IDs
    ids = pd.unique(df[['id_start', 'id_end']].values.ravel('K'))
    
    # Create an empty distance matrix
    distance_matrix = pd.DataFrame(np.inf, index=ids, columns=ids)
    
    # Fill the distance matrix with known distances
    for _, row in df.iterrows():
        from_id = row['id_start']
        to_id = row['id_end']
        distance = row['distance']
        
        # Update the distance matrix
        distance_matrix.at[from_id, to_id] = distance
        distance_matrix.at[to_id, from_id] = distance  # Ensure symmetry

    # Set diagonal values to 0
    np.fill_diagonal(distance_matrix.values, 0)

    # Use Floyd-Warshall algorithm to compute cumulative distances
    for k in ids:
        for i in ids:
            for j in ids:
                if distance_matrix.at[i, j] > distance_matrix.at[i, k] + distance_matrix.at[k, j]:
                    distance_matrix.at[i, j] = distance_matrix.at[i, k] + distance_matrix.at[k, j]

    # Return the distance matrix as df
    return distance_matrix

# Example usage
df = pd.read_csv('MapUp-DA-Assessment-2024\datasets\dataset-2.csv')
distance_matrix = calculate_distance_matrix(df)
# print(distance_matrix)



def unroll_distance_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Unroll a distance matrix to a DataFrame in the style of the initial dataset.

    Args:
        df (pandas.DataFrame): A distance matrix with IDs as both rows and columns.

    Returns:
        pandas.DataFrame: Unrolled DataFrame containing columns 'id_start', 'id_end', and 'distance'.
    """
    # Create an empty list to store the results
    results = []

    # Iterate over the rows and columns of the distance matrix
    for id_start in df.index:
        for id_end in df.columns:
            if id_start != id_end:  # Exclude same ID combinations
                distance = df.at[id_start, id_end]
                results.append({'id_start': id_start, 'id_end': id_end, 'distance': distance})

    # Convert the results list to a DataFrame
    unrolled_df = pd.DataFrame(results)

    return unrolled_df

# Example usage
# distance_matrix = calculate_distance_matrix(df)  # Assume this is the output of the previous function
unrolled_df = unroll_distance_matrix(distance_matrix)
# print(unrolled_df)


def find_ids_within_ten_percentage_threshold(df: pd.DataFrame, reference_id: int) -> pd.DataFrame:
    """
    Find all IDs whose average distance lies within 10% of the average distance of the reference ID.

    Args:
        df (pandas.DataFrame): DataFrame containing columns 'id_start', 'id_end', and 'distance'.
        reference_id (int): The reference ID to compare distances against.

    Returns:
        pandas.DataFrame: DataFrame with IDs whose average distance is within the specified percentage threshold
                          of the reference ID's average distance.
    """

    # Filter the DataFrame for the reference_id
    reference_distances = df[df['id_start'] == reference_id]
     
    
    if reference_distances.empty:
        return pd.DataFrame(columns=['id_start', 'average_distance'])

    # Calculate the average distance for the reference_id
    average_distance_ref = reference_distances['distance'].mean()
    
    # Calculate the 10% threshold
    lower_bound = average_distance_ref * 0.9
    upper_bound = average_distance_ref * 1.1

    # Calculate average distances for all id_start values
    average_distances = df.groupby('id_start')['distance'].mean().reset_index()
    average_distances.columns = ['id_start', 'average_distance']

    # Filter IDs within the threshold
    filtered_ids = average_distances[(average_distances['average_distance'] >= lower_bound) &  (average_distances['average_distance'] <= upper_bound)]


    return filtered_ids.sort_values(by='id_start')

# Example usage
# unrolled_df = unroll_distance_matrix(distance_matrix)  # Assume this is the output from the previous function
result = find_ids_within_ten_percentage_threshold(unrolled_df, reference_id=1001404)
# print(result)


def calculate_toll_rate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate toll rates for each vehicle type based on the unrolled DataFrame.

    Args:
        df (pandas.DataFrame): Input DataFrame containing columns 'id_start', 'id_end', and 'distance'.

    Returns:
        pandas.DataFrame: Input DataFrame with additional columns for toll rates by vehicle type.
    """
    # Define the rate coefficients for each vehicle type
    rates = {
        'moto': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6
    }
    
    # Calculate toll rates for each vehicle type
    for vehicle, rate in rates.items():
        df[vehicle] = df['distance'] * rate
    
    # Drop the distance column from the output
    #  df = df.drop(columns=['distance'])
    
    return df

toll_df = calculate_toll_rate(unrolled_df)
# print(toll_df)


def calculate_time_based_toll_rates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate time-based toll rates for different time intervals within a day.

    Args:
        df (pandas.DataFrame): Input DataFrame containing vehicle toll rates and distances.

    Returns:
        pandas.DataFrame: Updated DataFrame with time-based toll rates and additional columns.
    """
   # Define the days of the week and their corresponding start and end times
    time_config = {
        "Monday": (time(0, 0), time(10, 0)),
        "Tuesday": (time(10, 0), time(18, 0)),
        "Wednesday": (time(18, 0), time(23, 59, 59)),
        "Saturday": (time(0, 0), time(23, 59, 59)),
    }

    # Vehicle rates
    rates = {
        'moto': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6
    }


    # Create a list to store the results
    results = []
    
    # Iterate over each row in the DataFrame
    for index, row in df.iterrows():
        id_start = row['id_start']
        id_end = row['id_end']
        distance = row['distance']
        
        # Use original rates from the calculated tolls
        moto =  round(row['moto'], 2)
        car = round(row['car'], 2)
        rv = round(row['rv'], 2)
        bus = round(row['bus'], 2)
        truck =  round(row['truck'], 2)
        
        
        # Generate entries for the specified days
        for day, (start_time, end_time) in time_config.items():
            # Determine end_day based on the start_day
            if day == "Monday":
                end_day = "Friday"
            elif day == "Tuesday":
                end_day = "Saturday"
            elif day == "Wednesday":
                end_day = "Sunday"
            elif day == "Saturday":
                end_day = "Sunday"

            # Prepare the row
            row_result = {
                'id_start': int(id_start),
                'id_end': int(id_end),
                'distance': distance,
                'start_day': day,
                'start_time': start_time,
                'end_day': end_day,
                'end_time': end_time,
                'moto': moto,
                'car': car,
                'rv': rv,
                'bus': bus,
                'truck': truck
            }
            
           # Adjust toll rates based on the day
            if day == "Monday":
                row_result['moto'] = round(moto * 0.8, 2)  # 20% discount
                row_result['car'] = round(car * 0.8, 2)
                row_result['rv'] = round(rv * 0.8, 2)
                row_result['bus'] = round(bus * 0.8, 2)
                row_result['truck'] = round(truck * 0.8, 2)
            elif day == "Tuesday":
                row_result['moto'] = round(moto * 1.2, 2)  # 20% surcharge
                row_result['car'] = round(car * 1.2, 2)
                row_result['rv'] = round(rv * 1.2, 2)
                row_result['bus'] = round(bus * 1.2, 2)
                row_result['truck'] = round(truck * 1.2, 2)
            elif day == "Wednesday":
                row_result['moto'] = round(moto * 0.8, 2)
                row_result['car'] = round(car * 0.8, 2)
                row_result['rv'] = round(rv * 0.8, 2)
                row_result['bus'] = round(bus * 0.8, 2)
                row_result['truck'] = round(truck * 0.8, 2)
            elif day == "Saturday":
                row_result['moto'] = round(moto * 0.7, 2)  # 30% discount
                row_result['car'] = round(car * 0.7, 2)
                row_result['rv'] = round(rv * 0.7, 2)
                row_result['bus'] = round(bus * 0.7, 2)
                row_result['truck'] = round(truck * 0.7, 2)

            
            results.append(row_result)

    # Convert results to DataFrame
    result_df = pd.DataFrame(results)

    # Ensure the output columns are in the desired order
    output_columns = ['id_start', 'id_end', 'distance', 'start_day', 'start_time', 'end_day', 'end_time', 'moto', 'car', 'rv', 'bus', 'truck']
    
    return result_df[output_columns]

time_based_toll_df = calculate_time_based_toll_rates(toll_df)
print(time_based_toll_df)
