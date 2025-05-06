import re
from agent.plugins import PluginManager
from agent.openai_wrapper import OpenAIClient  # Optional

def parse_command(input_text: str, use_llm: bool = False):
    if use_llm:
        return OpenAIClient().parse_command(input_text)
    else:
        return regex_parse(input_text)  # Your handcrafted function

def regex_parse(text):
    # Example regex-based parsing for "deploy" commands
    match = re.search(r'deploy (kubernetes|aws|github)(.*)', text, re.IGNORECASE)
    if match:
        platform = match.group(1).lower()
        extra = match.group(2).strip()
        params = {}
        if "autoscaling" in extra:
            params["autoscaling"] = True
        if m := re.search(r'(\d+) nodes?', extra):
            params["nodes"] = int(m.group(1))
        return {"command": "deploy", "platform": platform, **params}
    return None
