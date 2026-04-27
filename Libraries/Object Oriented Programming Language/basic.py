class Student:
    
    def __init__(self,name,marks):
        self.marks = marks
        self.name = name
    
    def passed(self):
        return self.marks >=40
    
    def grade(self):
        
        if self.marks >= 80:
            return "A"
        
        elif self.marks >= 65:
            return "B"
        
        if self.marks >= 40:
            return "C"
        
        else:
            return "Fail"
        

s1 = Student("Shubham",95)
s2 = Student("Raj",90)
s3 = Student("Supriya",40)
s4 = Student("Sonali",35)

student_name = [s1,s2,s3,s4]

for i in student_name:
    if i.passed():
        print(f'Name : {i.name}, Grade : {i.grade()}, Status : {'Passed'}')
        
    else:
        print(f'Name : {i.name}, Grade : {i.grade()}, Status : {'Fail'}')