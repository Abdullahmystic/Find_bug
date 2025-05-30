# name, age, major = "Abdullah", 24, "CS"
#
# print(f"My name is {name}, I am {age} years old and my major is {major}.")
# import math
# import time
# import random
# from enum import nonmember

# height = 5.84
#
# print(round(height))

# name = input("Enter your name: ")
# age = int(input("Enter your age: "))
# major = input("What major are you in: ")
#
# age += 1
#
# print(f"Your name is {name}, you are {age} years old, and your major is {major}.")

# name = "Abdullah Sohail"

# print(name[9:])

# age = int(input("How old are you? "))
#
# if age >= 18 and age < 100:
#     print("You are an adult!")
# elif age < 18:
#     "You are still a child!"
# elif age >= 100:
#     print("You are century years old!")
# else:
#     print("You haven't been born yet!")

# x = 2
# for i in range(10+1):
#     print(f"{i} * {x} = {i*x}" )
#     i+=1

# rows = int(input("Enter the number of rows: "))
# columns = int(input("Enter the number of the columns: "))
# symbol = input("Enter the symbol: ")
#
# for i in range(rows):
#     for j in range(columns):
#         print(symbol, end="")
#     print()

# dinner = ["Burger", "Pizza", "shawrma"]
# drinks = ["coffee", "soda", "tea"]
# desserts = ["ice cream", "cake"]
#
# food = [dinner, drinks, desserts]
# print(food[0][1])

# for i in range(10, 0, -1):
#     print(i)
#     time.sleep(1)
# print("Happy Birthday!")

# dinner = {"soup", "chicken", "mutton"}
# utensils = {"fork", "spoon", "knife"}
#
# utensils.update(dinner)
#
# for i in utensils:
#     print(i, end=", ")

# def profile (name, age, gender, height, weight):
#     print(f"Your name is {name}.")
#     print(f"You are {age} years old.")
#     print(f"You are {gender}.")
#     print(f"You are {height}cm tall.")
#     print(f"You weight is {weight}kg.")
#
# profile("Abdullah", 24,"Male",178, "80")

# print(abs(int(input("Enter a number: "))))

# def add(*numb):
#     sum = 0
#     l1 = list(numb)
#     for i in l1:
#         sum += i
#     return sum
#
# x = add(1,3,5,15,6)
# print(x)

# try:
#     nominator = int(input("Enter a number to divide: "))
#     denominator = int(input("Enter a number to divide by: "))
#     result = nominator / denominator
#
# except ZeroDivisionError as a:
#     print(a)
#     print("You cant divide by zero.")
# except ValueError as x:
#     print(x)
#     print("Input must be a number.")
# except Exception as x:
#     print(x)
#     print("Something went wrong...")
# else:
#     print(round(result))
# finally:
#     print("I will always execute!~")

# import os
#
# path = "C:\\Users\\abdul\\OneDrive\\Desktop\\text.txt"
#
# if os.path.exists(path):
#     print("Path exists.")
# else:
#     print("Path does not exist.")

# with open('text.txt') as file:
#     print(file.read())
# print(file.closed)
#
# with open('C:\\Users\\abdul\\OneDrive\\Desktop\\text.txt') as file:
#     print(file.read())
#
# print(file.closed)
# text = "This is a new file\nIs this working."
# with open('newFile.txt', 'w') as file:
#     print(file.write(text))
# with open('newFile.txt') as file:
#     print(file.read())
#
# append = "\nAppending new data in the file."
# with open('newFile.txt', 'a') as file:
#     print(file.write(append))
#
# with open('newFile.txt') as file:
#     print(file.read())

# import os
#
# source = "text.txt"
# destination = "C:\\Users\\abdul\\OneDrive\\Desktop"
#
# try:
#     if os.path.exists("text.txt"):
#         print(f"{source} file already exists.")
#     else:
#         os.replace(source ,destination)
#         print(f" was moved")
# except FileNotFoundError:
#     print(f"{source} File doest not exist.")

# import random
# import time

#
# choices = ["rock", "paper", "scissors"]
#
# computer = random.choice(choices)
#
# answer = input("Would you like to play rock, paper, scissors. \n YES OR NO: ")
#
# user_input= input("Enter rock, paper or scissors: ")
#
# if "YES" == answer:
#     for i in choices:
#         print(i)
#         time.sleep(1)
#     print("\nShoot")
#     if computer == "rock":
#         print(f"\nComputer ROCK! vs user: {user_input}")
#     elif computer == "paper":
#         print(f"\nComputer: PAPER! vs user: {user_input}")
#     else:
#         print(f"\nComputer: Scissors! vs user: {user_input}")
# else:
#     print("Ok, Have a nice day.")

# from Car import Car
#
# car1 = Car("Ford", "Toyota", 1999, "Black")
# car2 = Car("Cultus", "Suzuki", 2001, "Silver")
#
# print(car1.model)
# print(car1.company)
# print(car1.year)
# print(car1.color)
# print("\n")
# print(car2.model)
# print(car2.company)
# print(car2.year)
# print(car2.color)
#
# car1.drive()
# car2.stop()

# name = input("Enter your username: ")
#
# if len(name) > 12 or len(name) <8:
#     print("Username should not be more then 12 characters.")
# elif name.find(" ") != -1:
#     print("Username should not have any spaces.")
# elif not name.isalpha():
#     print("Username should not have numbers or special characters.")
# else:
#     print(f"{name} Welcome!")

# import time
# give_time = int(input("Enter the time to set the timer in seconds: "))
# for i in range(give_time, 0, -1):
#     seconds = i % 60
#     minutes = int(i/60) % 60
#     hour = int( i / 3600)
#     print(f"{hour:02}:{minutes:02}:{seconds:02}")
#     time.sleep(1)
#
# print("TIME'S UP!")

# rows = int(input("Enter the rows: "))
# columns = int(input("Enter the columns: "))
# symbol = input("Enter the symbol: ")
#
# for x in range(rows):
#     for y in range(columns):
#         print(symbol,end="")
#     print()

# questions = ("What is the largest animal in the world: ",
#              "What is the highest mountain in the world: ",
#              "Who is the most famous person in the world: ",
#              "Which is the largest planet in the solar system: ")
# guesses = []
# options = (("A: Shark","B: Mouse","C: Whale","D: Octopus"),
#            ("A: K2","B: Mount Everest","C: Yellow stone","D: The great mount"),
#            ("A: Einstein","B: Trump","C: Prophet Muhammad","D: Nawaz"),
#            ("A: Earth","B: Venus","C: pluto","D: Jupiter"))
#
# answers = ("C","B", "C", "D")
# score= 0
# question_num = 0
#
# for question in questions:
#     print("--------------------------")
#     print(question)
#     for option in options[question_num]:
#         print(option)
#     guess = input("Enter the answer of this question: ").upper()
#     if guess == answers[question_num]:
#         print(f"{answers[question_num]} is CORRECT! ")
#         guesses.append(guess)
#         score += 1
#     else:
#         print(f"INCORRECT!. Correct Answer is {answers[question_num]}")
#
#     question_num +=1
# print("========================")
# print("         RESULT         ")
# print("========================")
#
# print("Answers: ", end="")
# for i in answers:
#     print(i, end=" ")
# print()
# print("Guesses: ", end="")
# for i in guesses:
#     print(i, end=" ")
# print()
# score = int(score /len(answers) *100)
#
# print(f"You score is {score}%.")

# cart = []
# total = 0
#
# menu = {
#     'Pizza slice':500,
#     'Chips':250,
#     'Popcorn':300,
#     'Soda':200,
#     'Juice':150
# }
# print("------------ Menu ------------")
# for key,value in menu.items():
#     print(f"{key:15}: {value}Rps")
# print("------------------------------")
#
# while True:
#     food = input("Enter the items in cart (q to quit): ").capitalize()
#     if food == "Q":
#         break
#     elif menu.get(food) is not None:
#         cart.append(food)
# print("------------ YOUR ORDER ------------")
# for food in cart:
#     total += menu.get(food)
#     print(food, end=" ")
# print()
# print("------------ TOTAL ------------")
# print(f"Your total is {total}Rps.")

# options = ("Paper","Rock","Scissors")
# running = True
# computer = random.choice(options)
# print("----------- LET THE GAMES BEGIN! -----------")
# while running:
#     player = None
#     while player not in options:
#         player = input("Enter the choice (Paper , Rock , Scissors): ").capitalize()
#         print("----------------------")
#         print(f"Computer: {computer}")
#         print(f"Player: {player}")
#         if player == computer:
#             print("IT'S A TIE!")
#             print("----------------------")
#         elif player == "Rock" and computer == "Scissors":
#             print("YOU WIN!")
#             print("----------------------")
#         elif player == "Paper" and computer == "Rock":
#             print("YOU WIN!")
#             print("----------------------")
#         elif player == "Scissors" and computer == "Paper":
#             print("YOU WIN!")
#             print("----------------------")
#         else:
#             print("YOU LOSE!")
#             print("----------------------")
#
#     if not input("Play Again (y/n): ").lower() == "y":
#         running = False

# print("\u25CF \u250C \u2500 \u2510 \u2502 \u2514 \u2518 ")
# ● ┌ ─ ┐ │ └ ┘
# "┌─────────────┐"
# "│             │"
# "│             │"
# "│             │"
# "│             │"
# "└─────────────┘"
# import random
# dice_art = {1:("┌───────────┐",
#                "│           │",
#                "│     ●     │",
#                "│           │",
#                "└───────────┘"),
#             2:("┌───────────┐",
#                "│  ●        │",
#                "│           │",
#                "│        ●  │",
#                "└───────────┘"),
#             3:("┌───────────┐",
#                "│  ●        │",
#                "│     ●     │",
#                "│        ●  │",
#                "└───────────┘"),
#             4:("┌───────────┐",
#                "│  ●     ●  │",
#                "│           │",
#                "│  ●     ●  │",
#                "└───────────┘"),
#             5:("┌───────────┐",
#                "│  ●     ●  │",
#                "│     ●     │",
#                "│  ●     ●  │",
#                "└───────────┘"),
#             6:("┌───────────┐",
#                "│   ●   ●   │",
#                "│   ●   ●   │",
#                "│   ●   ●   │",
#                "└───────────┘"),
#             }
# dice = int(random.choices(dice_art))

# dice = []
# total = 0
# num_of_dice = int(input("How many dice to roll: "))
#
# for die in range(num_of_dice):
#     dice.append(random.randint(1,6))

# for die in range(num_of_dice):
#     for line in dice_art.get(dice[die]):
#         print(line)

# for line in range(5):
#     for die in dice:
#         print(dice_art.get(die)[line],end="")
#     print()
#
# for die in dice:
#     total += die
# print(f"Total: {total}")

# def address(*args, **kwargs):
#     print("Your name is: ",end="")
#     for arg in args:
#         print(arg, end=" ")
#     print("\nYour address is: ",end="")
#     for key, value in kwargs.items():
#         print(f"{key}: {value}")
#
# address('Hafiz','Muhammad', 'Abdullah', 'Suhail',
#     House_no='211 Eden canal Villas',
#     City='Lahore',
#     state='Punjab',
#     Country='Pakistan'
# )

# characters = {"Hinata":'Milf',
#           "Tsunade":'Gilf',
#           "Kushina":'Milf',
#           "Boruto":'Shota',
#           "Jiraiya":'Dilf'}
# while True:
#     character = input("Enter the name of the female: ").capitalize()
#     if 'Q'!= character:
#        if character in characters:
#            print(f"{character} is a {characters[character]}")
#        else:
#            print(f"{character} does not exist.")
#     else:
#         break

# def func1():
#     x= 1
#     def fun2():
#         print(x)
#     fun2()
# func1()

# def show_balance(balance):
#     print(f"Your current balance is ${balance}")
#
# def deposit(balance):
#     amount = int(input("Enter the amount you want to enter: "))
#
#     if amount < 0:
#         print("================ERROR==================")
#         print("The amount you entered is insufficient.")
#         print("=======================================")
#         deposit(balance)
#     elif amount > 0:
#         balance += amount
#         return balance
#
# def withdraw(balance):
#     amount = int(input("Enter the amount you want to withdraw: "))
#
#     if amount < 0:
#         print("================ERROR==================")
#         print("The amount you entered is insufficient.")
#         print("=======================================")
#         withdraw(balance)
#     elif amount > balance:
#         print("==============ERROR==============")
#         print("The amount is insufficient funds.")
#         print("=================================")
#         withdraw(balance)
#     else:
#         balance -= amount
#         return balance
#
#
# def main():
#
#     balance = 1000
#     is_running = True
#
#     while is_running:
#         print("=================================")
#         print("Enter the choices between (1-4): ")
#         print("1.Check the balance: ")
#         print("2.Deposit money in the bank: ")
#         print("3.Withdraw money from the bank: ")
#         print("4.Exit: ")
#         option = input("Enter the choice: ")
#         print("=================================")
#
#         if option == '1':
#             show_balance(balance)
#         elif option == '2':
#            balance = deposit(balance)
#         elif option == '3':
#             balance = withdraw(balance)
#         elif option == '4':
#             is_running = False
#
# if __name__ == '__main__':
#     main()

# import random
#
# def spin_row(row):
#     wheel = ['🔔','🍋','🍒','🪙','💰']
#
#     return [random.choice(wheel) for _ in range(3)]
#
# def print_row(row):
#     print("*************")
#     print(" | ".join(row))
#     print("*************")
#
# def payment_method(bet, row):
#     if row[0] == row[1] == row[2]:
#         if row[0] == '🔔':
#             return bet * 2
#
#         elif row[0] == '🍋':
#             return bet * 4
#
#         elif row[0] == '🍒':
#             return bet * 5
#
#         elif row[0] == '🪙':
#             return bet * 10
#
#         elif row[0] == '💰':
#             return bet * 20
#
#     return 0
#
#
# def main():
#     balance = 100
#
#     print("************************************")
#     print("******Welcome to python slots*******")
#     print("Symbols: 🔔,🍋,🍒,🪙,💰")
#     print("************************************")
#
#
#     while balance > 0:
#         print(f"Your current balance is ${balance}")
#         bet = input("Enter your bet: ")
#         if not bet.isdigit():
#             print("Please enter a valid number.")
#             continue
#
#         bet = int(bet)
#
#         if bet > balance:
#             print("Insufficient funds")
#             continue
#
#         if bet <= 0:
#             print("bet must be greater then 0.")
#             continue
#         balance -= bet
#
#         row = []
#         row = spin_row(row)
#         print("\nSpinning...")
#         print_row(row)
#         payout = payment_method(bet, row)
#
#         if payout > 0:
#             print(f"Your amount is: {payout}$")
#         else:
#             print("You lost\n")
#
#         balance += payout
#
#         play_again = input("Would you like to play again: (Y/N) ").upper()
#
#         if play_again != "Y":
#             break
#
# if '__main__' == __name__:
#     main()

# import random
# import string
#
# chars = " " + string.digits + string.punctuation + string.ascii_letters
#
# chars = list(chars)
#
# keys = chars.copy()
# random.shuffle(keys)

# print(f"chars: {chars}")
# print(f"keys: {keys}")

# original_message = input("Enter the message: ")
# cypher_message = ""
#
# for letter in original_message:
#     index = chars.index(letter)
#     cypher_message += keys[index]
#
# print(f"\noriginal_message : {original_message}")
# print(f"encrypted_message: {cypher_message} ")
#
# cypher_message = input("\nEnter the message to decrypt: ")
# original_message = ""
#
#
# for letter in cypher_message:
#     index = keys.index(letter)
#     original_message += chars[index]
#
# print(f"\nencrypted_message: {cypher_message} ")
# print(f"original_message : {original_message}")

import random
import string

chars = string.digits + string.ascii_letters + string.punctuation + " "
chars = list(chars)
keys = chars.copy()

random.shuffle(keys)

original_text = input("Enter the text: ")
cypher_text = ""

for letter in original_text:
    index = chars.index(letter)
    cypher_text += keys[index]

print(f"original text: {original_text}")
print(f"Encrypted text: {cypher_text}")

cypher_text = input("Enter the encrypted text: ")
original_text = ""


for letter in cypher_text:
    index = keys.index(letter)
    original_text += chars[index]

print(f"Encrypted text: {cypher_text}")
print(f"Decrypted text: {original_text}")