# class Person:
#     def __init__(self, fname, lname):
#         self.firstname = fname
#         self.lastname = lname 
    
#     def printname(self):
#         print(self.firstname, self.lastname)
    
# class Location:
#     def __init__(self, country):
#         self.country = country 
    
#     def print_location(self):
#         print(self.country)

# class Student(Person, Location):
#     pass 

# x = Student('Manish', 'Khurmi', 'India')
# x.printname()
# x.print_location()


class Person:
  def __init__(self, fname, lname):
    self.firstname = fname
    self.lastname = lname

  def printname(self):
    print(self.firstname, self.lastname)

class Location:
  def __init__(self, country):
    self.country = country

  def print_location(self):
    print(self.country)

class Student(Person, Location):
  pass  # No additional methods needed here

# Create a Student object
x = Student('Manish', 'Khurmi', 'India')  # Notice we added the country argument

# Call the inherited methods
x.printname()
x.print_location()