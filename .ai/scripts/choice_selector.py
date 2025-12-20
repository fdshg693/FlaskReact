"""
Interactive choice selector script.

Displays a message and a list of choices that users can navigate with arrow keys.
Includes an optional "Other" choice for free-text input.

Usage:
    python choice_selector.py "Your message" "Choice1" "Choice2" "Choice3"
    python choice_selector.py --no-other "Your message" "Choice1" "Choice2"
"""

import platform
import sys

if platform.system() == "Windows":
    import msvcrt

    def get_arrow_key():
        """Read arrow key input on Windows."""
        if msvcrt.kbhit():
            key = msvcrt.getch()
            if key == b"\xe0":  # Arrow key prefix on Windows
                arrow = msvcrt.getch()
                if arrow == b"H":  # Up arrow
                    return "up"
                elif arrow == b"P":  # Down arrow
                    return "down"
            elif key == b"\r":  # Enter key
                return "enter"
        return None
else:
    import termios
    import tty

    def get_arrow_key():
        """Read arrow key input on Unix/macOS."""
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            key = sys.stdin.read(1)
            if key == "\x1b":  # Escape sequence
                next_char = sys.stdin.read(1)
                if next_char == "[":
                    arrow = sys.stdin.read(1)
                    if arrow == "A":  # Up arrow
                        return "up"
                    elif arrow == "B":  # Down arrow
                        return "down"
            elif key == "\r":  # Enter key
                return "enter"
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return None


def clear_lines(n):
    """Move cursor up and clear lines."""
    for _ in range(n):
        sys.stdout.write("\033[A\033[K")
    sys.stdout.flush()


def display_choices(message, choices, selected_index):
    """Display the message and choices with the current selection highlighted."""
    print(message)
    print()
    for i, choice in enumerate(choices):
        if i == selected_index:
            print(f"  > {choice}")
        else:
            print(f"    {choice}")
    sys.stdout.flush()


def select_choice(message, choices):
    """Main selection loop using arrow keys."""
    selected_index = 0
    total_lines = len(choices) + 2  # message + blank line + choices

    # Initial display
    display_choices(message, choices, selected_index)

    while True:
        key = get_arrow_key()

        if key == "up":
            selected_index = (selected_index - 1) % len(choices)
            clear_lines(total_lines)
            display_choices(message, choices, selected_index)
        elif key == "down":
            selected_index = (selected_index + 1) % len(choices)
            clear_lines(total_lines)
            display_choices(message, choices, selected_index)
        elif key == "enter":
            return selected_index, choices[selected_index]


def main():
    args = sys.argv[1:]

    if not args:
        print(
            'Usage: python choice_selector.py [--no-other] "message" "choice1" "choice2" ...'
        )
        sys.exit(1)

    # Check for --no-other flag
    include_other = True
    if args[0] == "--no-other":
        include_other = False
        args = args[1:]

    if len(args) < 2:
        print("Error: Please provide a message and at least one choice.")
        sys.exit(1)

    message = args[0]
    choices = args[1:]

    # Add "Other" option if enabled
    other_label = "そのほか"
    if include_other:
        choices = choices + [other_label]

    # Run selection
    index, selected = select_choice(message, choices)

    print()

    # Handle "Other" selection
    if include_other and selected == other_label:
        user_input = input("自由入力してください: ")
        print()
        print(f"入力内容: {user_input}")
    else:
        print(f"選択された項目: {selected}")


if __name__ == "__main__":
    main()
