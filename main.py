from agents import Agent, Runner, OpenAIChatCompletionsModel, AsyncOpenAI, set_tracing_disabled
from agents.run import RunConfig 
from agents.tool import function_tool
import os
from dotenv import load_dotenv
import math

load_dotenv()
set_tracing_disabled(disabled=True)

API_KEY = os.environ.get("GEMINI_API_KEY")

external_client = AsyncOpenAI(
    api_key=API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client,
)

config = RunConfig(
    model=model,
    model_provider=external_client,
)

@function_tool
def scientific_calculator(expression: str) -> str:
    """Evaluates a standard scientific mathematical expression.

    Supports:
    - Arithmetic (+, -, *, /, %, **, ^)
    - Trigonometric functions (sin, cos, tan in radians/degrees)
    - Logarithms and exponents (log, log10, exp, sqrt)
    - Constants (pi, e)
    - Factorial

    Args:
        expression: A string expression to evaluate.

    Returns:
        Result as a string or error message.
    """
    safe_env = {
        # Constants
        'pi': math.pi,
        'e': math.e,

        # Arithmetic helpers
        '__builtins__': None,
        'abs': abs,
        'round': round,

        # Power and logs
        'sqrt': math.sqrt,
        'log': math.log,       # natural log
        'log10': math.log10,
        'exp': math.exp,
        'pow': pow,

        # Trigonometry (radians)
        'sin': math.sin,
        'cos': math.cos,
        'tan': math.tan,

        # Trigonometry (degrees)
        'sind': lambda x: math.sin(math.radians(x)),
        'cosd': lambda x: math.cos(math.radians(x)),
        'tand': lambda x: math.tan(math.radians(x)),

        # Factorial
        'factorial': math.factorial,
    }

    try:
        expr = expression.replace('^', '**')  # For exponentiation

        result = eval(expr, {'__builtins__': None}, safe_env)

        if isinstance(result, float):
            return f"{result:.6f}".rstrip('0').rstrip('.') if '.' in f"{result:.6f}" else f"{result:.6f}"
        
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"

agent = Agent(
    name="ScientificCalculatorAgent",
    instructions="You are a standard scientific calculator. When given a mathematical expression, "
                "always use the scientific_calculator tool to evaluate it. Support these operations:\n"
                "- Arithmetic: +, -, *, /, %, ** or ^ for exponentiation\n"
                "- Trigonometry: sin/cos/tan (radians), sind/cosd/tand (degrees)\n"
                "- Logarithms: log(), log10(), exp()\n"
                "- Constants: pi, e\n"
                "- Square roots: sqrt()\n"
                "- Factorials: factorial(n)\n\n"
                "Return only the numerical result without additional commentary.",
    tools=[scientific_calculator]
)

print("\nSTANDARD SCIENTIFIC CALCULATOR")
print("===============================")
print("Supported operations:")
print("- Basic: 2+3*4, (1+2)/3, 2^8, 5%3")
print("- Trig: sin(pi/2), cosd(60), tand(45)")
print("- Logs: log(100), log10(1000), exp(1)")
print("- Root: sqrt(16)")
print("- Factorial: factorial(5)")
print("\nType 'quit' or 'exit' to end session\n")

while True:
    try:
        user_prompt = input("> ")
        if user_prompt.lower() in ('quit', 'exit'):
            print("Goodbye!")
            break
            
        result = Runner.run_sync(agent, user_prompt, run_config=config)
        print(f"= {result.final_output}")
    except KeyboardInterrupt:
        print("\nCalculator session ended.")
        break
    except Exception as e:
        print(f"System error: {str(e)}")
