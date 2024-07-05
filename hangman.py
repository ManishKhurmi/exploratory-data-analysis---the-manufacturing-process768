# Hangman
import random

class Hangman:
    # Task 1
    def __init__(self, word_list, num_lives):

        '''
        This class is used to represent the Hangman Game.

        Attributes:
        word (str): the random word chosen from the word_list
        word_guessed (list): a list to store the correct guesses e.g. creates ['_','_','_'] for the word 'cat'
        num_letters (int): the number of unique letters not guessed yet 
        num_lives (int): the number of lives the player has at the start of the game 
        word_list (list): A list of words from which the game will randomly choose one for the player to guess. 
                          For example, ['mango', 'lychee', 'banana'].
        self.list_of_guesses (list): the list of the players' guesses
        '''
      
        self.word = random.choice(word_list) 
        self.word_guessed = len(self.word) * ['_'] 
        self.num_letters = len(set(self.word))
        self.num_lives = num_lives 
        self.word_list = word_list 
        self.list_of_guesses = [] 

    # Task 2
    def check_guess(self, guess):
        '''
        This function checks if the players guessed letter is in the randomly generated word.

        This function processes a player's guess by converting it to lowercase and checking if it
        is present in the word to be guessed. If the guess is correct, it updates the word_guessed 
        list to reveal the guessed letters and decreases the count of remaining unique letters by
        1. If the guess is incorrect, it decreases the players number of lives by 1.

        Attribues:
        guess (str): The letter guessed by the player
        '''
        guess = guess.lower() 
        
        if guess in self.word: 
            print(f'Good guess! {guess} is in the word')
            for letter in self.word:
                if letter == guess:
                    self.word_guessed[self.word.index(letter)] = guess
            self.num_letters -= 1 
        else: 
            
            self.num_lives -= 1
            print(f'Sorry, {guess} is not in the word')
            print(f'You have {self.num_lives} lives left')

            
    # Task 2
    def ask_for_input(self):
        '''
        Prompts the user to guess a letter and provide feedback on the outcome.

        This function continously prompts the player to guess a letter until a valid guess is 
        provied. It validates the input to ensure it is an input that hasn't been guessed 
        before. If the guess is valid, it updates the list of guesses and checks if the 
        guess is in the word using the check_guess() method.
        '''
        while True: 
            guess = input('Guess a Letter:')
            
            if len(guess) != 1 or not guess.isalpha():
                print('Invalid letter. Please, enter a single alphabetical character')
            elif guess in self.list_of_guesses:
                print('You already tried that letter!')
            else: 
                self.list_of_guesses.append(guess) 
                self.check_guess(guess)
                break

def play_game(word_list):
    '''
    Execute the main game loop for Hangman.

    This function intialises the Hangman game with a given list of words and a fixed number of lives. It continously 
    prompts the player for guesses and provides feedback on their progress. The game ends when the player either 
    guesses all the letters in the word or runs out of lives.


    Attributes:
    word_list (list): A list of words from which the game will randomly choose one for the player to guess. 
                      For example, ['mango', 'lychee', 'banana'].
    '''

    num_lives = 5
    game = Hangman(word_list, num_lives)

    while True:
        if game.num_lives == 0:
            print('You lost!')
            break
        elif game.num_letters == 0:
            print('Congratulations. You won the game!')
            break
        else:
            game.ask_for_input()

# Use the help() to find out information about the Hangman Game
#help(Hangman)

# Testing play_game()
play_game(word_list = ['apples', 'bananas','cherries'])