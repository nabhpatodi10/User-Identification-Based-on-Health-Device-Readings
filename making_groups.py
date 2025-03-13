import pandas as pd
import random
import math

# Load the existing dataset
df = pd.read_csv('blood_pressure_glucose_data_100_people_3months.csv')

# Get unique PersonIDs
all_person_ids = df['PersonID'].unique()

# Separate PersonIDs based on Blood Glucose readings
zero_glucose_people = []
nonzero_glucose_people = []

for person_id in all_person_ids:
    person_data = df[df['PersonID'] == person_id]
    total_glucose = person_data['BloodGlucose'].sum()
    if total_glucose == 0:
        zero_glucose_people.append(person_id)
    else:
        nonzero_glucose_people.append(person_id)

# Check counts
num_zero = len(zero_glucose_people)
num_nonzero = len(nonzero_glucose_people)

print(f"Total people with zero glucose readings: {num_zero}")
print(f"Total people with non-zero glucose readings: {num_nonzero}")

# Generate group sizes between 2 and 7 that sum up to 100
def generate_group_sizes(total_people, min_size=2, max_size=7):
    group_sizes = []
    remaining_people = total_people
    while remaining_people >= min_size:
        max_possible_size = min(max_size, remaining_people - (min_size * ((remaining_people - min_size) // max_size)))
        group_size = random.randint(min_size, max_possible_size)
        group_sizes.append(group_size)
        remaining_people -= group_size
    # If any people are left ungrouped, add them to the last group
    if remaining_people > 0:
        group_sizes[-1] += remaining_people
    return group_sizes

group_sizes = generate_group_sizes(100)
random.shuffle(group_sizes)  # Shuffle to randomize group sizes

print(f"Generated group sizes: {group_sizes}")
print(f"Total groups formed: {len(group_sizes)}")

# Shuffle the lists to randomize assignments
random.shuffle(zero_glucose_people)
random.shuffle(nonzero_glucose_people)

groups = []
zero_index = 0
nonzero_index = 0

for idx, size in enumerate(group_sizes):
    group = []
    # Assign one zero glucose person
    if zero_index < num_zero:
        group.append(zero_glucose_people[zero_index])
        zero_index += 1
    else:
        # If no zero glucose people left, take from nonzero and mark for later adjustment
        pass

    # Assign one nonzero glucose person
    if nonzero_index < num_nonzero:
        group.append(nonzero_glucose_people[nonzero_index])
        nonzero_index += 1
    else:
        # If no nonzero glucose people left, take from zero glucose and mark for later adjustment
        pass

    remaining_slots = size - len(group)

    # Calculate remaining zero and nonzero people
    remaining_zero = num_zero - zero_index
    remaining_nonzero = num_nonzero - nonzero_index
    total_remaining_people = remaining_zero + remaining_nonzero

    if total_remaining_people == 0 and remaining_slots > 0:
        # All people have been assigned, but group size not met
        # Assign randomly from already assigned people (or adjust group sizes)
        pass

    # Determine how many zero and nonzero people to assign based on remaining counts
    if remaining_slots > 0:
        zero_needed = min(math.ceil(remaining_slots / 2), remaining_zero)
        nonzero_needed = remaining_slots - zero_needed

        # Adjust if not enough people left in one category
        if nonzero_needed > remaining_nonzero:
            zero_needed += nonzero_needed - remaining_nonzero
            nonzero_needed = remaining_nonzero
        elif zero_needed > remaining_zero:
            nonzero_needed += zero_needed - remaining_zero
            zero_needed = remaining_zero

        # Add zero glucose people
        for _ in range(zero_needed):
            if zero_index < num_zero:
                group.append(zero_glucose_people[zero_index])
                zero_index += 1

        # Add nonzero glucose people
        for _ in range(nonzero_needed):
            if nonzero_index < num_nonzero:
                group.append(nonzero_glucose_people[nonzero_index])
                nonzero_index += 1

    groups.append(group)

# Output the groups
for i, group in enumerate(groups, 1):
    print(f"Group {i} (Size {len(group)}): {group}")
