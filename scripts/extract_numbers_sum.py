import re
import sys


def extract_numbers(text: str) -> list[float]:
    return [float(match) for match in re.findall(r"-?\d+(?:\.\d+)?", text)]


def main() -> None:
    text = sys.stdin.read()
    numbers = extract_numbers(text)
    total = sum(numbers)

    print("Numbers:", numbers)
    print("Sum:", total)


if __name__ == "__main__":
    main()