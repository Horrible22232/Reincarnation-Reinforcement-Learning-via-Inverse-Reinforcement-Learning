# from neroRRL.environments.wrapper import wrap_environment
# from neroRRL.utils.yaml_parser import YamlParser
# import time

# # Load environment, model, evaluation and training parameters
# config = YamlParser("./configs/crafter.yaml").get_config()
# # Load environment
# env = wrap_environment(config["environment"], 0, realtime_mode = False, record_trajectory = False)
# # Reset environment
# vis_obs, _ = env.reset()
# # Start the timer
# start_time = time.time()

# *_, = env.step([env.action_space.sample()])

# # Calculate the elapsed time
# elapsed_time = time.time() - start_time

# # Print the elapsed time
# print(f"Execution time: {elapsed_time} seconds")


import threading

def wait_and_execute(lst, function):
    with lst.condition:
        while not lst.is_filled():
            lst.condition.wait()  # Wait until the list is filled

        # Execute the function
        function()

class FilledList:
    def __init__(self):
        self.items = []
        self.condition = threading.Condition()

    def add_item(self, item):
        with self.condition:
            self.items.append(item)
            self.condition.notify_all()  # Notify all waiting threads

    def is_filled(self):
        return len(self.items) >= 5  # Adjust the condition as needed


# Usage example
def my_function():
    print("List is filled. Executing the function!")

lst = FilledList()

# Create the thread
thread = threading.Thread(target=wait_and_execute, args=(lst, my_function))
thread.start()

# Add items to the list
for i in range(5):
    lst.add_item(i)

# Wait for the thread to finish
thread.join()