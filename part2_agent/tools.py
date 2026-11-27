"""Five small tools with made-up data, their JSON schemas, and a dispatcher.

The lookup tools (weather, stock price) return invented values, so the model
cannot answer from memory and must call the tool.
"""
import ast
import datetime as dt
import json
import operator

# ----------------------------------------------------------------------------
# Fake data
# ----------------------------------------------------------------------------
WEATHER = {  # city -> (temperature in Celsius, condition)  -- invented values
    "colombo": (31, "humid and sunny"),
    "kandy": (24, "light rain"),
    "paris": (12, "overcast"),
    "tokyo": (17, "clear"),
    "nairobi": (22, "partly cloudy"),
    "oslo": (-3, "snow"),
    "lima": (19, "foggy"),
    "toronto": (5, "windy"),
    "cairo": (34, "sunny"),
    "sydney": (26, "thunderstorms"),
    "berlin": (8, "drizzle"),
    "mumbai": (29, "hazy"),
}

STOCKS = {  # invented tickers -> price in USD
    "ZORB": 142.37,
    "QLXT": 58.12,
    "NIMB": 311.05,
    "VEXA": 23.90,
    "TORQ": 87.44,
    "PLUM": 4.75,
    "KRYO": 1204.50,
    "SOLN": 66.60,
}

# canonical unit -> aliases
UNIT_ALIASES = {
    "km": ["km", "kilometer", "kilometers", "kilometre", "kilometres", "kms"],
    "mi": ["mi", "mile", "miles"],
    "kg": ["kg", "kilogram", "kilograms", "kgs", "kilo", "kilos"],
    "lb": ["lb", "lbs", "pound", "pounds"],
    "C": ["c", "celsius", "°c", "degc", "centigrade"],
    "F": ["f", "fahrenheit", "°f", "degf"],
    "m": ["m", "meter", "meters", "metre", "metres"],
    "ft": ["ft", "foot", "feet"],
    "L": ["l", "liter", "liters", "litre", "litres"],
    "gal": ["gal", "gallon", "gallons"],
    "cm": ["cm", "centimeter", "centimeters", "centimetre", "centimetres"],
    "in": ["in", "inch", "inches"],
}
_ALIAS_TO_UNIT = {a: u for u, aliases in UNIT_ALIASES.items() for a in aliases}

_LINEAR = {  # (from, to) -> factor
    ("km", "mi"): 0.621371,
    ("kg", "lb"): 2.204623,
    ("m", "ft"): 3.280840,
    ("L", "gal"): 0.264172,
    ("cm", "in"): 0.393701,
}


def canon_unit(u):
    if u is None:
        return None
    return _ALIAS_TO_UNIT.get(str(u).strip().lower())


# ----------------------------------------------------------------------------
# Tools
# ----------------------------------------------------------------------------
_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.USub: operator.neg, ast.UAdd: operator.pos,
}


def _safe_eval(node):
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        left, right = _safe_eval(node.left), _safe_eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("exponent too large")
        return _OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("unsupported expression")


def eval_expression(expression):
    expr = str(expression).replace("^", "**").replace("×", "*").replace("÷", "/").replace(",", "")
    return _safe_eval(ast.parse(expr, mode="eval"))


def calculator(expression):
    try:
        value = eval_expression(expression)
    except Exception as e:  # noqa: BLE001
        return {"error": f"could not evaluate expression: {e}"}
    return {"expression": str(expression), "result": round(float(value), 4)}


def convert_units(value, from_unit, to_unit):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return {"error": "value must be a number"}
    f, t = canon_unit(from_unit), canon_unit(to_unit)
    if f is None or t is None:
        return {"error": f"unknown unit(s): {from_unit}, {to_unit}"}
    if f == t:
        out = value
    elif (f, t) == ("C", "F"):
        out = value * 9 / 5 + 32
    elif (f, t) == ("F", "C"):
        out = (value - 32) * 5 / 9
    elif (f, t) in _LINEAR:
        out = value * _LINEAR[(f, t)]
    elif (t, f) in _LINEAR:
        out = value / _LINEAR[(t, f)]
    else:
        return {"error": f"cannot convert {from_unit} to {to_unit}"}
    return {"value": value, "from_unit": f, "to_unit": t, "result": round(out, 2)}


def get_weather(city):
    key = str(city).strip().lower()
    if key not in WEATHER:
        return {"error": f"no weather data for '{city}'"}
    temp, cond = WEATHER[key]
    return {"city": key.title(), "temperature_c": temp, "condition": cond}


def get_stock_price(ticker):
    key = str(ticker).strip().upper()
    if key not in STOCKS:
        return {"error": f"unknown ticker '{ticker}'"}
    return {"ticker": key, "price_usd": STOCKS[key]}


_DATE_FORMATS = ["%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%B %d, %Y", "%b %d, %Y", "%d %B %Y", "%d %b %Y"]


def parse_date(s):
    s = str(s).strip()
    for fmt in _DATE_FORMATS:
        try:
            return dt.datetime.strptime(s, fmt).date()
        except ValueError:
            pass
    return None


def days_between(start_date, end_date):
    a, b = parse_date(start_date), parse_date(end_date)
    if a is None or b is None:
        return {"error": "dates must be in YYYY-MM-DD format"}
    return {"start_date": a.isoformat(), "end_date": b.isoformat(), "days": (b - a).days}


TOOLS = {
    "calculator": calculator,
    "convert_units": convert_units,
    "get_weather": get_weather,
    "get_stock_price": get_stock_price,
    "days_between": days_between,
}

