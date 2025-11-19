# nuScenes Semantic Captioning System - Setup Guide

## Overview

This system implements an agentic pipeline for generating high-quality semantic captions for nuScenes driving scenes, inspired by the AgentInstruct paper (Mitra et al., 2024).

## Architecture

The pipeline consists of 4 layers:

1. **Content Transformation Flow**: Parallel agents process multi-modal inputs
   - CameraAgent: Vision-Language Model for camera images
   - LiDARAgent: Processes 3D point clouds
   - SceneGraphAgent: Converts annotations to structured scene graphs
   - BEVFusionAgent: Fuses multi-modal information
   - CrossModalAgent: Facilitates information exchange

2. **Seed Features Generation**: Multiple LLM agents extract features focusing on different aspects (objects, spatial relations, dynamics, safety, scene structure)

3. **Features Refinement**: Suggester + Editor agents refine the features

4. **Caption Generation**: Produces structured JSON captions and answers nuScenes-MQA questions

## Installation

### 1. Prerequisites

```bash
# Tested with Python 3.10.12

# Create virtual environment
python -m venv captioning_env
source captioning_env/bin/activate  # On Windows: captioning_env\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. nuScenes Dataset

1. Download nuScenes dataset from [nuScenes website](https://www.nuscenes.org/download). A free account is required.
2. Download v1.0-mini (10 scenes, ~3.5GB) for testing or full v1.0-trainval (850 scenes, ~365GB)
3. Extract to a directory, e.g., `/data/nuscenes/`

Directory structure should be:
```
/data/nuscenes/
├── v1.0-mini/
│   ├── samples/
│   ├── sweeps/
│   ├── maps/
│   └── v1.0-mini/
└── v1.0-trainval/ (optional)
```

### 4. Azure OpenAI Setup

1. **Get Azure OpenAI Access**:
   - Sign up for [Azure for Students](https://azure.microsoft.com/en-us/free/students/)
   - Or use regular Azure account

2. **Create Azure OpenAI Resource**:
   ```bash
   # In Azure Portal:
   # 1. Create Resource > Azure OpenAI
   # 2. Note down: Endpoint URL and API Key
   # 3. Deploy models: gpt-4o-mini (for testing) and optionally gpt-4o
   ```

3. **Configure Environment Variables**:
   
   Create `.env` file:
   ```bash
   AZURE_OPENAI_API_KEY=your_api_key_here
   AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
   AZURE_OPENAI_API_VERSION=2024-02-15-preview
   
   NUSCENES_DATAROOT=/data/nuscenes/v1.0-mini
   NUSCENES_VERSION=v1.0-mini
   ```

## Quick Start

### 1. Test with Mock Data (No nuScenes Required)

1. Ensure `USE_MOCK` is set to `True` in `src/mock_test.py`
2. Run the script:
```bash
python src/mock_test.py
```

## 2. Test with Real nuScenes Data

1. Change `USE_MOCK` to `False` in `src/mock_test.py`
2. Run the script:
```bash
python src/mock_test.py
```

### 3. Run Ablation Study

```bash
python src/evaluation_framework.py
```

## Cost Estimation (Azure OpenAI)

Using **gpt-4o-mini** for testing:
- Input: ~$0.15 per 1M tokens
- Output: ~$0.60 per 1M tokens

**Estimated costs per scene** (6 cameras + LiDAR + annotations):
- Layer 1 (5 agents): ~15K tokens input, ~2K tokens output ≈ $0.0034
- Layer 2 (5 agents): ~5K tokens input, ~1K tokens output ≈ $0.0014
- Layer 3 (2 agents): ~3K tokens input, ~1K tokens output ≈ $0.0011
- Layer 4 (1 agent): ~2K tokens input, ~0.5K tokens output ≈ $0.0006

**Total per scene: ~$0.0065**

With $100 credit, you can process approximately **15,000 scenes** using gpt-4o-mini!

## Recommended Testing Strategy

### Phase 1: Pipeline Development (Budget: $10)
- Use gpt-4o-mini
- Test with 5-10 scenes
- Debug and refine pipeline
- Verify outputs look reasonable

### Phase 2: Ablation Studies (Budget: $30)
- Process 50-100 scenes
- Test all modality configurations
- Measure performance differences
- Identify optimal configuration

### Phase 3: Full Evaluation (Budget: $40)
- Process 500-1000 scenes
- Generate comprehensive captions
- Evaluate against nuScenes-MQA
- Compare with baselines

### Phase 4: Final Production (Budget: $20)
- Switch to gpt-4o for quality
- Process selected challenging scenes
- Generate final dataset release

## References

- AgentInstruct Paper: Mitra et al., 2024
- nuScenes Dataset: [nuscenes.org](https://www.nuscenes.org/)
- Azure OpenAI: [docs.microsoft.com/azure/ai-services/openai](https://docs.microsoft.com/azure/ai-services/openai)
