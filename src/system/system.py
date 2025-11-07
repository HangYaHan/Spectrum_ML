import os
import json

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..\\config.json")

class Config:
    def __init__(self, config_path):
        self.config_path = config_path
        self._load_config()

    def _load_config(self):
        with open(self.config_path, "r") as f:
            self._config = json.load(f)

    def get(self, key):
        keys = key.split(".")
        value = self._config
        for k in keys:
            value = value[k]
        return value

    def set(self, key, value):
        keys = key.split(".")
        config = self._config
        for k in keys[:-1]:
            config = config[k]
        config[keys[-1]] = value
        self._save_config()

    def _save_config(self):
        with open(self.config_path, "w") as f: 
            json.dump(self._config, f, indent=2)

def main_loop():
    # Backwards-compatible wrapper: instantiate SystemShell and run
    shell = SystemShell()
    shell.run()


class SystemShell:
    """Interactive command shell for the system.

    - Use `run()` to start a blocking prompt loop.
    - Use `handle_command(cmd_str)` to process a single command programmatically.
    """
    def __init__(self):
        pass

    def run(self):
        """Start the interactive loop (blocking)."""
        while True:
            try:
                user_input = input("> ")
            except (EOFError, KeyboardInterrupt):
                print("\nExiting main loop. Goodbye!")
                break

            if not user_input:
                continue

            parts = user_input.strip().split()
            cmd = parts[0].lower()

            if cmd == 'q':
                print("Exiting main loop. Goodbye!")
                break
            elif cmd == 'help':
                self.print_help()
            elif cmd == 'config':
                self._handle_config_command(user_input)
            elif cmd == 'run':
                # support: 'run' and 'run -t'
                if len(parts) == 1:
                    print("[placeholder] run invoked (normal mode)")
                elif len(parts) == 2 and parts[1] == '-t':
                    print("[placeholder] run invoked (test mode -t)")
                else:
                    print("Error: Unknown run options. Supported: -t")
            elif cmd == 'cls':
                # clear the screen
                os.system('cls' if os.name == 'nt' else 'clear')
            elif cmd == 'pxana':
                print("[placeholder] pxana invoked (pixel analysis) - implementation pending")
            else:
                print(f"Unknown command: {user_input}")

    def handle_command(self, cmd_str: str):
        """Handle a single command string (non-blocking). Useful for programmatic use or tests."""
        if not cmd_str:
            return

        parts = cmd_str.strip().split()
        cmd = parts[0].lower()

        if cmd == 'q':
            return 'quit'
        if cmd == 'help':
            return self.print_help()
        if cmd == 'config':
            return self._handle_config_command(cmd_str)
        if cmd == 'run':
            if len(parts) == 1:
                return "[placeholder] run invoked (normal mode)"
            if len(parts) == 2 and parts[1] == '-t':
                return "[placeholder] run invoked (test mode -t)"
            return "Error: Unknown run options. Supported: -t"
        if cmd == 'cls':
            os.system('cls' if os.name == 'nt' else 'clear')
            return None
        if cmd == 'pxana':
            return "[placeholder] pxana invoked (pixel analysis) - implementation pending"
        return f"Unknown command: {cmd_str}, enter 'help' for available commands."

    def _handle_config_command(self, user_input: str):
        args = user_input.split()
        if len(args) == 1:
            return view_config()
        if args[1] == "-v":
            if len(args) == 3 and args[2].startswith("--"):
                module = args[2][2:]
                return view_config(module)
            else:
                return view_config()
        if args[1] == "-s":
            if len(args) == 4:
                key = args[2]
                value = args[3]
                return set_config(key, value)
            else:
                print("Error: Invalid number of arguments for setting config.")
                return None
        print("Error: Unknown config command.")
        return None

    # expose existing helpers as instance methods for convenience
    def print_help(self):
        return print_help()


def print_help():
    print("1. help - Show this help menu")
    print("2. config - View the entire config file")
    print("   config -v --[module] - View a specific module in the config file")
    print("   config -s <key> <value> - Set a specific key in the config file")
    print("3. run - Start dataprocess pipeline and training")
    print("   run -t - Start the main run and save temporary files")
    print("4. cls - Clear the console")
    print("5. pxana - Start pixel analysis module")

def view_config(module=None):
    config = load_config()
    if module:
        if module in config:
            print(json.dumps(config[module], indent=2))
        else:
            print(f"Error: Module '{module}' not found in the config file.")
    else:
        print(json.dumps(config, indent=2))

def set_config(key, value):
    config = load_config()
    keys = key.split(".")
    current = config

    # Traverse to the correct key
    for k in keys[:-1]:
        if k in current:
            current = current[k]
        else:
            print(f"Error: Key '{key}' does not exist in the config file.")
            return

    # Validate and set the value
    if keys[-1] in current:
        try:
            # Attempt to cast the value to the same type as the existing value
            current[keys[-1]] = type(current[keys[-1]])(value)
            save_config(config)
            print(f"Parameter '{key}' updated to {value}.")
        except ValueError:
            print(f"Error: Value for '{key}' must be of type {type(current[keys[-1]]).__name__}.")
    else:
        print(f"Error: Key '{key}' does not exist in the config file.")

def load_config():
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)

def save_config(config):
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)

def handle_args(args):
    if args.h:
        print_help()
    elif args.v:
        view_config(args.module)
    elif args.s:
        set_config(args.s[0], args.s[1])
    else:
        print_help()

def parse_and_execute():
    import argparse

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-h", action="store_true", help="Show help message.")
    parser.add_argument("-v", action="store_true", help="View the config file.")
    parser.add_argument("--module", type=str, help="View a specific module in the config file.")
    parser.add_argument("-s", nargs=2, metavar=("key", "value"), help="Set a specific key in the config file.")

    args = parser.parse_args()
    handle_args(args)