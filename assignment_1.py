def lower_triangle(n):
    print("Lower Triangular Pattern:")
    for i in range(1, n + 1):
        print("* " * i)
    print()

def upper_triangle(n):
    print("Upper Triangular Pattern:")
    for i in range(n, 0, -1):
        spaces = n - i
        print("  " * spaces + "* " * i)
    print()

def pyramid(n):
    print("Pyramid Pattern:")
    for i in range(1, n + 1):
        spaces = n - i
        print(" " * spaces + "* " * i)
    print()

try:
    user_input = input("Enter the number of rows (default is 5): ")
    rows = int(user_input) if user_input.strip() else 5
except ValueError:
    rows = 5

lower_triangle(rows)
upper_triangle(rows)
pyramid(rows)
