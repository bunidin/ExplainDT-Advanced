import random

def generate_random_instance(length):
    sequence = [random.choice([0, 1]) for _ in range(length)]
    return sequence
    
def generate_random_partial_instance(length):
    sequence = [random.choice([0, 1, '?']) for _ in range(length)]
    return sequence

def main():
    length = int(input("Enter the length of the sequence: "))
    
    # Generate and print the random full sequence
    random_sequence = generate_random_instance(length)
    formatted_sequence = ','.join(map(str, random_sequence))
    print(f'Full sequence: ({formatted_sequence})')
    
    # Generate and print the random partial sequence
    random_partial_sequence = generate_random_partial_instance(length)
    formatted_partial_sequence = ','.join(map(str, random_partial_sequence))
    print(f'Partial sequence: ({formatted_partial_sequence})')

if __name__ == "__main__":
    main()
