class Node:
    def __init__(self, data):
        self.data = data
        self.next = None


class LinkedList:
    def __init__(self):
        self.head = None

    def add_node(self, data):
        new_node = Node(data)
        if not self.head:
            self.head = new_node
            return
        current = self.head
        while current.next:
            current = current.next
        current.next = new_node

    def print_list(self):
    
        current = self.head
        if not current:
            print("List is empty.")
            return
        while current:
            print(current.data, end=" -> ")
            current = current.next
        print("None")

    def delete_nth_node(self, n):
    
        if not self.head:
            raise Exception("Cannot delete from an empty list.")

        if n <= 0:
            raise Exception("Invalid index. Index must be 1 or greater.")

        if n == 1:
            print(f"Deleting node at position {n}: {self.head.data}")
            self.head = self.head.next
            return

        current = self.head
        prev = None
        count = 1

        while current and count < n:
            prev = current
            current = current.next
            count += 1

        if not current:
            raise Exception(f"Index {n} is out of range.")

        print(f"Deleting node at position {n}: {current.data}")
        prev.next = current.next


if __name__ == "__main__":
    ll = LinkedList()

    print("Enter elements to add to the linked list (type 'done' to finish):")
    while True:
        user_input = input("Enter value: ")
        if user_input.lower() == 'done':
            break
        try:
            value = int(user_input)
            ll.add_node(value)
        except ValueError:
            print("Please enter a valid integer or 'done'.")

    print("\nYour Linked List:")
    ll.print_list()

    while True:
        delete_input = input("\nEnter the position (1-based index) of the node to delete (type 'exit' to quit): ")
        if delete_input.lower() == 'exit':
            break
        try:
            index = int(delete_input)
            ll.delete_nth_node(index)
            print("Updated Linked List:")
            ll.print_list()
        except Exception as e:
            print("Error:", e)
