# plugins/base.py
class Plugin:
    def __init__(self, name):
        self.name = name
    
    def execute(self):
        pass

class AWSPlugin(Plugin):
    def execute(self):
        pass
