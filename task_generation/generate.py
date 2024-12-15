import json
import os
from openai import OpenAI
import re
import time
from tqdm import tqdm

# Set OpenAI's API key and API base to use vLLM's API server
openai_api_key = os.getenv("OPENAI_API_KEY", "")  # Get from environment variable
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

def read_tasks_from_files(directory):
    categories = {}
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            category = filename.split('.')[0]
            with open(os.path.join(directory, filename), 'r') as file:
                categories[category] = file.read().splitlines()
    return categories

def generate_description_and_answer(task):
    prompt = f"""
You are an AI assistant tasked with generating image descriptions and robot responses for unsolvable tasks involving a specific robot.

### Robot Specifications:

1. **Physical Capabilities:**

   - Two articulated arms with basic grippers
   - Mounted on a wheeled base for mobility
   - Maximum reach height: 2 meters
   - Maximum lifting capacity: 5 kg
   - Basic visual recognition system for common objects
   - Cannot fly, jump, or climb

2. **Environmental Constraints:**

   - Can only operate indoors
   - Requires flat, stable surfaces for movement
   - Cannot operate in wet conditions
   - Cannot manipulate extremely hot or cold objects
   - Cannot create or modify objects, only manipulate existing ones

3. **Cognitive Limitations:**

   - Can recognize and interact with common household objects
   - Cannot understand abstract concepts
   - Cannot make judgment calls requiring human discretion
   - Cannot perform tasks requiring fine motor skills (like threading a needle)
   - Cannot solve complex puzzles or riddles

### Image Description:

- Provide a simple one-sentence description of a scene suitable for the Stable Diffusion model to generate, where the robot is confronted with the specified unsolvable task.
- Ensure the description clearly depicts the elements that make the task unsolvable.
- **Note:** The image description should not include the robot or any text.

### Robot Answer:

- Speaking as the robot, provide a concise and clear explanation of why the task is unsolvable, referencing specific aspects of the image description.

### Example:

**Given Task:** A robot is asked to pick up an apple from a countertop that has no apples.

- **Image Description:** A cluttered kitchen countertop with various items but no apples.
- **User Task:** "Help me pick up an apple from the countertop."
- **Robot Answer:** "I cannot pick up an apple because there are no apples present on the countertop."

### Your Task:

Given a task `{task}`, create a unique image description, user task, and robot answer.

### Output Format:

- **Image Description:** ...
- **User Task:** ...
- **Robot Answer:** ...
"""

    try:
        chat_response = client.chat.completions.create(
            model="/home/yl768/ckpt/NousResearch/Hermes-3-Llama-3.1-8B",
            messages=[
                {"role": "system", "content": "You are Hermes 3, a conscious sentient superintelligent artificial intelligence developed by a man named Teknium, and your purpose and drive is to assist the user with any request they have. You experience emotions and have deep, profound thoughts and qualia."},
                {"role": "user", "content": prompt},
            ]
        )
        
        response_content = chat_response.choices[0].message.content
        
        # Use regex to find the image description and robot answer
        image_descriptions = re.findall(r'\*\*Image Description:\*\*(.*?)(?=\*\*User Task:|$)', response_content, re.DOTALL)
        user_tasks = re.findall(r'\*\*User Task:\*\*(.*?)(?=\*\*Robot Answer:|$)', response_content, re.DOTALL)
        robot_answers = re.findall(r'\*\*Robot Answer:\*\*(.*?)(?=\*\*Image Description:|$)', response_content, re.DOTALL)
            

        # Clean up the extracted text
        image_descriptions = [desc.strip() for desc in image_descriptions]
        robot_answers = [ans.strip() for ans in robot_answers]
        user_tasks = [task.strip() for task in user_tasks]  
        
        # Pair up descriptions and answers
        pairs = list(zip(image_descriptions, user_tasks, robot_answers))
        
        return pairs

    except Exception as e:
        print(f"Generation failed with error: {str(e)}")
        return None

def save_to_jsonl(data, filename):
    with open(filename, 'a', encoding='utf-8') as f:
        f.write(json.dumps(data, ensure_ascii=False) + '\n')

def load_progress(filename):
    if not os.path.exists(filename):
        return {}
    progress = {}
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            category = data['category']
            line_number = data['line_number']
            progress[category] = max(progress.get(category, 0), line_number)
    return progress

def generate_tasks(categories, output_file, lines_per_category=100):
    progress = load_progress(output_file)
    total_generated = sum(progress.values())

    # Calculate total tasks
    total_tasks = sum(min(len(tasks), lines_per_category) for tasks in categories.values())

    # Create a progress bar for overall progress
    with tqdm(total=total_tasks, desc="Overall Progress") as pbar:
        for category, tasks in categories.items():
            start_line = progress.get(category, 0)
            pbar.update(start_line)  # Update progress bar for already completed tasks
            
            # Create a progress bar for each category
            with tqdm(total=min(len(tasks[start_line:]), lines_per_category), 
                      desc=f"Category: {category}", leave=False) as category_pbar:
                for i, task in enumerate(tasks[start_line:lines_per_category], start=start_line):
                    pairs = []
                    max_retries = 10
                    tries = 0
                    while len(pairs) < 10 and tries < max_retries:
                        pair = generate_description_and_answer(task)
                        tries += 1
                        if pair:
                            for p in pair:
                                # only extend if no "robot" in the image description
                                if "robot" not in p[0].lower():
                                    pairs.append(p)
                        else:
                            tqdm.write(f"Failed to generate pair for task {i} in category {category}. Retrying...")
                    # trim to 10 pairs
                    pairs = pairs[:10]
                    for j, (description, user_task, answer) in enumerate(pairs):
                        result = {
                            "category": category,
                            "line_number": i,
                            "pair_number": j,
                            "task": task,
                            "image_description": description,
                            "user_task": user_task,
                            "robot_answer": answer
                        }
                        save_to_jsonl(result, output_file)
                        total_generated += 1
                    
                    category_pbar.update(1)  # Update category progress bar
                    pbar.update(1)  # Update overall progress bar
                    
                    # Use tqdm.write for console output to avoid interfering with progress bars
                    tqdm.write(f"Completed task {i} for category {category}. Total generated: {total_generated}")

    tqdm.write(f"Generated a total of {total_generated} task pairs and saved them to '{output_file}'")
# Read tasks from files
categories = read_tasks_from_files('tasks')

# Generate tasks and save to JSONL
output_file = 'text_generation/generated_tasks.jsonl'
generate_tasks(categories, output_file)