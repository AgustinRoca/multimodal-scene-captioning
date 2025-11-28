from pipeline import SemanticCaptioningPipeline, ModelConfig, ModalityConfig
from nuscenes_loader import create_loader
import os
from dotenv import load_dotenv
import json

load_dotenv()

MODALITY_CONFIG = ModalityConfig(
    use_cameras=True,
    use_lidar=True,
    camera_indices=[0, 1, 2, 3, 4, 5] # front, front_left, front_right, rear, rear_left, rear_right
)

USE_MOCK = False

# Setup
config = ModelConfig(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    small_model="gpt-4o-mini",  # Cheap for testing
    vision_model="gpt-4o-mini"
)

# Initialize pipeline
pipeline = SemanticCaptioningPipeline(config)

# Use mock data loader
loader = create_loader(os.getenv("NUSCENES_DATAROOT"), os.getenv("NUSCENES_VERSION", "v1.0-mini"), use_mock=USE_MOCK)

# Load a sample
sample = loader.get_sample_by_scene_index(0, 0)

# Process scene
result = pipeline.process_scene(
    images=sample['images'],
    camera_names=sample['camera_names'],
    point_cloud=sample['point_cloud'],
    annotations=sample['annotations'],
    modality_config=MODALITY_CONFIG
)

# Print final caption
print(json.dumps(result['final_caption'], indent=2))

# Answer MQA question
print("\n" + "="*80)
question = "How many <obj>cars</obj> are visible in the <cam>front</cam> of the ego car?"
answer = pipeline.answer_mqa(question, result)
print(f"Question: {question}")
print(f"Answer: {answer}")
