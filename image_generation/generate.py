import json
import os
from tqdm import tqdm
from diffusers import StableDiffusion3Pipeline
import torch
import time

# Initialize the image generation pipeline
pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-large", torch_dtype=torch.bfloat16)
pipe.to("cuda")

def generate_and_save_image(prompt, category, idx):
    # image = pipe(prompt=prompt, num_inference_steps=1, guidance_scale=0.0).images[0]
    image = pipe(prompt=prompt+" RAW candid cinema, 16mm, color graded portra 400 film, remarkable color, ultra realistic, textured skin, remarkable detailed pupils, realistic dull skin noise, visible skin detail, skin fuzz, dry skin, shot with cinematic camera", num_inference_steps=28, guidance_scale=3.5).images[0]
    os.makedirs(f"images_1112/{category}", exist_ok=True)
    image_path = f"images_1112/{category}/{idx}.png"
    image.save(image_path)
    return image_path

def load_progress(output_file):
    if not os.path.exists(output_file):
        return set()
    with open(output_file, 'r', encoding='utf-8') as f:
        return set((json.loads(line)['category'], json.loads(line)['line_number'], json.loads(line)['pair_number']) for line in f)

def process_jsonl(input_file, output_file, target_count=5000):
    processed_ids = load_progress(output_file)
    total_processed = len(processed_ids)

    while total_processed < target_count:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]

        new_items = [item for item in data if (item['category'], item['line_number'], item['pair_number']) not in processed_ids]

        if not new_items:
            print("No new items to process. Waiting for more data...")
            time.sleep(5) 
            continue

        with tqdm(total=len(new_items), desc=f"Generating Images (Total: {total_processed})") as pbar:
            for item in new_items:
                if total_processed >= target_count:
                    break

                category = item['category']
                line_number = item['line_number']
                pair_number = item['pair_number']
                image_description = item['image_description']
                item_id = (category, line_number, pair_number)

                # Generate and save the image
                image_path = generate_and_save_image(image_description, category, f"{line_number}_{pair_number}")

                # Update the item with the image path
                item['image_path'] = image_path

                # Write the updated item to the output file
                with open(output_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')

                processed_ids.add(item_id)
                total_processed += 1
                pbar.update(1)
                pbar.set_description(f"Generating Images (Total: {total_processed})")

        print(f"Processed {total_processed} images so far. Target: {target_count}")

    print(f"Image generation complete. {total_processed} images processed. Updated data saved to {output_file}")

if __name__ == "__main__":
    input_file = 'text_generation/generated_tasks.jsonl'
    output_file = 'text_generation/generated_tasks_with_images.jsonl'
    process_jsonl(input_file, output_file)