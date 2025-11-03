import argparse
from venv import logger
from src.system import system
from src.system import log

def interactive_mode():
    while True:
        user_input = input("Enter a command ('q' to quit): ")
        if user_input.lower() == 'q':
            print("Exiting interactive mode. Goodbye!")
            break
        elif user_input.lower() == 'help':
            system.print_help()
        else:
            print(f"Unknown command: {user_input}")

if __name__ == "__main__":
    logger1 = log.get_logger("system")
    logger2 = log.get_logger("test")
    logger1.info("Program started.")
    interactive_mode()
    logger2.info("This is a test message.")
    logger1.info("Program ended.")