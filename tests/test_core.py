# tests/test_core.py
def test_agent_initialization():
    from agent.core import AgentCore
    assert AgentCore() is not None
    
def test_execute_command():
    from agent.core import AgentCore
    agent = AgentCore()
    agent.execute_command('some_command')
    assert agent.context.get('command') == 'some_command'
