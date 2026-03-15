"""Default tools for CLI Agent."""

import datetime
import random

from cli_agent.core.tools import Tool


def get_current_time() -> str:
    """Get the current date and time."""
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")


def get_random_number(min_val: int = 1, max_val: int = 100) -> int:
    """Generate a random number within a range."""
    return random.randint(min_val, max_val)


def calculate(expression: str) -> str:
    """Evaluate a mathematical expression safely."""
    try:
        # Only allow safe operations
        allowed_names = {"abs": abs, "round": round, "max": max, "min": min, "sum": sum}
        code = compile(expression, "<string>", "eval")
        # Check that only allowed operations are used
        for name in code.co_names:
            if name not in allowed_names and not name.isdigit():
                raise ValueError(f"Disallowed name: {name}")
        result = eval(code, {"__builtins__": {}}, allowed_names)
        return str(result)
    except Exception as e:
        return f"Error: {e}"


def get_weather(location: str) -> dict:
    """Get weather information for a location (mock)."""
    # This is a mock implementation
    conditions = ["sunny", "cloudy", "rainy", "snowy", "windy"]
    temp = random.randint(-5, 35)
    condition = random.choice(conditions)

    return {
        "location": location,
        "temperature_c": temp,
        "temperature_f": temp * 9 // 5 + 32,
        "condition": condition,
        "humidity": random.randint(30, 90),
        "updated_at": datetime.datetime.now().isoformat(),
        "note": "This is mock data for demonstration purposes",
    }


def search_web(query: str, num_results: int = 3) -> list:
    """Search the web for information (mock)."""
    # This is a mock implementation
    return [
        {
            "title": f"Result {i + 1} for '{query}'",
            "url": f"https://example.com/search?q={query}&page={i + 1}",
            "snippet": f"This is a mock search result {i + 1} about {query}...",
        }
        for i in range(min(num_results, 5))
    ]


def save_note(title: str, content: str) -> str:
    """Save a note to a file."""
    filename = f"note_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"Title: {title}\n")
        f.write(f"Date: {datetime.datetime.now().isoformat()}\n")
        f.write("-" * 40 + "\n")
        f.write(content)
    return f"Note saved to {filename}"


def get_system_info() -> dict:
    """Get system information."""
    import platform
    import sys

    return {
        "platform": platform.platform(),
        "python_version": sys.version,
        "processor": platform.processor(),
        "machine": platform.machine(),
        "current_time": datetime.datetime.now().isoformat(),
    }


def get_default_tools() -> list[Tool]:
    """Get list of default tools."""
    return [
        Tool.from_function(get_current_time, description="Get the current date and time"),
        Tool.from_function(
            get_random_number,
            description="Generate a random number between min and max values",
        ),
        Tool.from_function(calculate, description="Evaluate a mathematical expression"),
        Tool.from_function(
            get_weather,
            description="Get weather information for a location (returns mock data)",
        ),
        Tool.from_function(
            search_web,
            description="Search the web for information (mock results for demo)",
        ),
        Tool.from_function(save_note, description="Save a note to a text file"),
        Tool.from_function(get_system_info, description="Get system information"),
    ]
