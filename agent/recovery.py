# agent/recovery.py
class Recovery:
    @staticmethod
    def handle_error(error):
        print(f"Error occurred: {error}")
        # Retry logic or ask user for intervention
