"""
Scene Graph Agent with True Hierarchical Scene Graph Structure
Builds tree-based scene graphs from object annotations using Pydantic
"""

from typing import Any, Dict, Optional, List, Literal
from agents import BaseAgent
import numpy as np
import json
from pydantic import BaseModel, Field
from collections import defaultdict


# ============================================================================
# Pydantic Models for Scene Graph Structure
# ============================================================================

class SceneObject(BaseModel):
    """Individual object in the scene"""
    id: str = Field(description="Unique object identifier")
    category: str = Field(description="Object category (car, pedestrian, etc.)")
    subcategory: str = Field(description="More specific type (sedan, adult, etc.)")
    position_x: float = Field(description="X position in meters")
    position_y: float = Field(description="Y position in meters")
    position_z: float = Field(description="Z position in meters")
    distance_to_ego: float = Field(description="Distance from ego vehicle in meters")
    direction: str = Field(description="Direction from ego (front, back, left, right)")
    state: str = Field(description="Object state (moving, stopped, parked)")
    visibility: str = Field(description="Visibility level (high, medium, low)")


class RoadElement(BaseModel):
    """Road-related element"""
    element_type: str = Field(description="Type of road element (lane, marking, sign)")
    description: str = Field(description="Description of the element")
    location: str = Field(description="Location relative to ego")


class LaneInfo(BaseModel):
    """Lane information"""
    lane_count: int = Field(description="Number of visible lanes")
    lane_type: str = Field(description="Lane type (highway, urban, intersection)")
    ego_lane_position: str = Field(description="Ego vehicle position (left, center, right)")
    lane_markings: List[str] = Field(description="Visible lane markings")


class RoadStructure(BaseModel):
    """Road structure and layout"""
    road_type: str = Field(description="Type of road (highway, urban street, intersection)")
    lanes: LaneInfo = Field(description="Lane information")
    road_elements: List[RoadElement] = Field(description="Road signs, markings, infrastructure")
    surface_condition: str = Field(description="Road surface condition")


class SidewalkArea(BaseModel):
    """Sidewalk area with pedestrians and objects"""
    has_sidewalk: bool = Field(description="Whether sidewalk is visible")
    pedestrians: List[SceneObject] = Field(description="Pedestrians on sidewalk")
    static_objects: List[SceneObject] = Field(description="Static objects (benches, trash cans)")
    location: str = Field(description="Sidewalk location (left, right, both)")


class TrafficParticipants(BaseModel):
    """Active traffic participants on the road"""
    vehicles: List[SceneObject] = Field(description="Vehicles on the road")
    cyclists: List[SceneObject] = Field(description="Bicycles and motorcycles")
    vulnerable_road_users: List[SceneObject] = Field(description="Pedestrians crossing or near road")


class StaticInfrastructure(BaseModel):
    """Static infrastructure elements"""
    barriers: List[SceneObject] = Field(description="Barriers and guardrails")
    traffic_cones: List[SceneObject] = Field(description="Traffic cones")
    construction: List[SceneObject] = Field(description="Construction equipment")
    other: List[SceneObject] = Field(description="Other static objects")


class EnvironmentContext(BaseModel):
    """Environmental context"""
    lighting: str = Field(description="Lighting conditions (day, night, dusk, dawn)")
    weather: str = Field(description="Weather conditions (clear, rain, fog)")
    visibility_overall: str = Field(description="Overall visibility (good, moderate, poor)")
    location_type: str = Field(description="Location type (urban, highway, residential)")


class SpatialZone(BaseModel):
    """Spatial zone around ego vehicle"""
    zone_name: str = Field(description="Zone identifier (front_close, left_medium, etc.)")
    objects: List[SceneObject] = Field(description="Objects in this zone")
    is_clear: bool = Field(description="Whether zone is clear of obstacles")
    criticality: str = Field(description="Safety criticality (high, medium, low)")


class HierarchicalSceneGraph(BaseModel):
    """
    Complete hierarchical scene graph
    
    Structure:
    Scene
    ├── Environment (lighting, weather, location type)
    ├── Road Structure
    │   ├── Lanes (count, type, markings)
    │   ├── Road Elements (signs, markings)
    │   └── Traffic Participants (vehicles, cyclists, pedestrians on road)
    ├── Sidewalk Areas
    │   ├── Pedestrians
    │   └── Static Objects (benches, trash cans)
    ├── Static Infrastructure (barriers, cones, construction)
    ├── Spatial Zones (front_close, front_medium, left_close, etc.)
    └── Safety Critical Elements
    """
    scene_summary: str = Field(description="Brief overall scene description")
    environment: EnvironmentContext = Field(description="Environmental conditions")
    road_structure: RoadStructure = Field(description="Road layout and structure")
    traffic_participants: TrafficParticipants = Field(description="Active road users")
    sidewalk_areas: SidewalkArea = Field(description="Sidewalk and pedestrian areas")
    static_infrastructure: StaticInfrastructure = Field(description="Static objects and barriers")
    spatial_zones: List[SpatialZone] = Field(description="Spatial zones around ego vehicle")
    safety_critical_elements: List[str] = Field(description="Safety-critical observations")
    total_objects: int = Field(description="Total number of detected objects")


# ============================================================================
# Scene Graph Agent
# ============================================================================

class SceneGraphAgent(BaseAgent):
    """
    Builds true hierarchical scene graphs with tree structure
    """
    
    def __init__(self, client, model: str, agent_name: str):
        super().__init__(client, model, agent_name)
        
        # Spatial zones definition
        self.spatial_zones = {
            'front_close': (0, 10, 'front'),
            'front_medium': (10, 30, 'front'),
            'front_far': (30, 50, 'front'),
            'left_close': (0, 10, 'left'),
            'left_medium': (10, 30, 'left'),
            'right_close': (0, 10, 'right'),
            'right_medium': (10, 30, 'right'),
            'back_close': (0, 10, 'back'),
            'back_medium': (10, 30, 'back'),
        }
    
    def process(self, annotations: List[Dict], context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Process object annotations into hierarchical scene graph
        
        Args:
            annotations: List of nuScenes object annotations
            context: Optional context from other agents
        """        
        # Step 1: Parse annotations into structured objects
        scene_objects = self._parse_annotations(annotations)
        
        # Step 2: Categorize objects by scene graph hierarchy
        categorized = self._categorize_objects(scene_objects)
        
        # Step 3: Build spatial zones
        spatial_zones = self._build_spatial_zones(scene_objects)
        
        # Step 4: Generate structured scene graph using LLM
        scene_graph = self._generate_scene_graph(
            categorized, spatial_zones, annotations, context
        )
        
        # Step 5: Generate human-readable summary
        summary = self._generate_summary(scene_graph)
        
        return {
            "agent": self.agent_name,
            "modality": "scene_graph",
            "scene_graph": scene_graph.model_dump(),
            "observations": summary
        }
    
    def _parse_annotations(self, annotations: List[Dict]) -> List[Dict]:
        """Parse nuScenes annotations into simplified object dicts"""
        objects = []
        
        for i, ann in enumerate(annotations):
            # Extract position
            pos = ann.get('translation', [0, 0, 0])
            
            # Calculate distance and direction
            distance = np.sqrt(pos[0]**2 + pos[1]**2)
            angle = np.arctan2(pos[1], pos[0]) * 180 / np.pi
            angle = (angle + 360) % 360
            
            # Determine direction
            if 45 <= angle < 135:
                direction = "front"
            elif 135 <= angle < 225:
                direction = "left"
            elif 225 <= angle < 315:
                direction = "back"
            else:
                direction = "right"
            
            # Parse category
            category = ann.get('category_name', 'unknown').lower()
            for prefix in ['vehicle.', 'human.pedestrian.', 'movable_object.', 'static_object.']:
                category = category.replace(prefix, '')
            
            # Determine state
            velocity = ann.get('velocity', None)
            if velocity is not None:
                try:
                    # Velocity is typically [vx, vy] (2D)
                    if isinstance(velocity, (list, tuple)) and len(velocity) >= 2:
                        vx, vy = velocity[0], velocity[1]
                        if vx is not None and vy is not None:
                            speed = np.sqrt(vx**2 + vy**2)
                            state = "moving" if speed > 0.5 else "stopped"
                        else:
                            state = "stopped"
                    else:
                        state = "stopped"
                except (TypeError, IndexError, ValueError):
                    state = "stopped"
            else:
                state = "stopped"
            
            # Parse visibility
            vis_token = str(ann.get('visibility_token', ''))
            if '80' in vis_token or '100' in vis_token:
                visibility = "high"
            elif '40' in vis_token or '60' in vis_token:
                visibility = "medium"
            else:
                visibility = "low"
            
            objects.append({
                'id': f"obj_{i}",
                'category': category,
                'position': pos,
                'distance': distance,
                'direction': direction,
                'state': state,
                'visibility': visibility,
                'attributes': ann.get('attribute_tokens', [])
            })
        
        return objects
    
    def _categorize_objects(self, objects: List[Dict]) -> Dict[str, List[Dict]]:
        """Categorize objects by their role in the scene hierarchy"""
        categorized = {
            'vehicles': [],
            'cyclists': [],
            'pedestrians': [],
            'barriers': [],
            'traffic_cones': [],
            'construction': [],
            'other': []
        }
        
        for obj in objects:
            cat = obj['category']
            
            if 'car' in cat or 'truck' in cat or 'bus' in cat or 'trailer' in cat:
                categorized['vehicles'].append(obj)
            elif 'bicycle' in cat or 'motorcycle' in cat:
                categorized['cyclists'].append(obj)
            elif 'pedestrian' in cat or 'adult' in cat or 'child' in cat:
                categorized['pedestrians'].append(obj)
            elif 'barrier' in cat:
                categorized['barriers'].append(obj)
            elif 'cone' in cat:
                categorized['traffic_cones'].append(obj)
            elif 'construction' in cat:
                categorized['construction'].append(obj)
            else:
                categorized['other'].append(obj)
        
        return categorized
    
    def _build_spatial_zones(self, objects: List[Dict]) -> Dict[str, List[Dict]]:
        """Organize objects into spatial zones"""
        zones = {zone_name: [] for zone_name in self.spatial_zones.keys()}
        
        for obj in objects:
            distance = obj['distance']
            direction = obj['direction']
            
            # Assign to appropriate zone
            for zone_name, (min_dist, max_dist, zone_dir) in self.spatial_zones.items():
                if direction == zone_dir and min_dist <= distance < max_dist:
                    zones[zone_name].append(obj)
                    break
        
        return zones
    
    def _generate_scene_graph(self, categorized: Dict, spatial_zones: Dict,
                             annotations: List[Dict], context: Optional[Dict]) -> HierarchicalSceneGraph:
        """Generate hierarchical scene graph using LLM with structured output"""
        
        system_prompt = """You are an expert at building hierarchical scene graphs for autonomous driving.

Create a tree-structured scene graph organizing the scene into:
1. Environment: lighting, weather, location type
2. Road Structure: lanes, markings, road elements
3. Traffic Participants: vehicles, cyclists, pedestrians on/near road
4. Sidewalk Areas: pedestrians on sidewalk, static objects
5. Static Infrastructure: barriers, cones, construction
6. Spatial Zones: objects organized by distance and direction from ego
7. Safety Critical: important safety observations

For each object, provide:
- id, category, subcategory
- position (x, y, z coordinates)
- distance_to_ego, direction
- state (moving/stopped/parked)
- visibility (high/medium/low)

CRITICAL INSTRUCTIONS:
- Include EVERY piece of information available - no summarization
- Be exhaustive and thorough - longer captions with more detail are better
- Don't say "various objects" or "several vehicles" - name each one specifically
- Include all numerical data (distances, counts, positions)
- Write as if you're describing the scene to someone who can't see it"""

        # Prepare input data summary
        object_summary = f"""
Total objects: {len(annotations)}

By category:
- Vehicles: {len(categorized['vehicles'])}
- Cyclists: {len(categorized['cyclists'])}
- Pedestrians: {len(categorized['pedestrians'])}
- Barriers: {len(categorized['barriers'])}
- Traffic cones: {len(categorized['traffic_cones'])}
- Construction: {len(categorized['construction'])}

Spatial distribution:
- Front close (<10m): {len(spatial_zones.get('front_close', []))}
- Front medium (10-30m): {len(spatial_zones.get('front_medium', []))}
- Left close (<10m): {len(spatial_zones.get('left_close', []))}
- Right close (<10m): {len(spatial_zones.get('right_close', []))}

Object details:
{json.dumps([{
    'id': obj['id'],
    'category': obj['category'],
    'position': obj['position'],
    'distance': round(obj['distance'], 1),
    'direction': obj['direction'],
    'state': obj['state'],
    'visibility': obj['visibility']
} for obj in categorized['vehicles'][:5] + categorized['pedestrians'][:5]], indent=2)}
... (showing sample, {len(annotations)} total)
"""

        user_prompt = f"""Build a hierarchical scene graph from this driving scene:

{object_summary}

Create a complete scene graph with all hierarchical levels filled."""

        if context:
            user_prompt += f"\n\nAdditional context from other sensors:\n{json.dumps(context, indent=2)[:500]}"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            response = self.call_llm(
                messages, 
                temperature=0.4,
                response_format=HierarchicalSceneGraph
            )
            return response
            
        except Exception as e:
            print(f"  ⚠️  Error generating scene graph: {e}")
            # Return minimal fallback
            return HierarchicalSceneGraph(
                scene_summary="Error generating scene graph",
                environment=EnvironmentContext(
                    lighting="unknown",
                    weather="unknown",
                    visibility_overall="unknown",
                    location_type="unknown"
                ),
                road_structure=RoadStructure(
                    road_type="unknown",
                    lanes=LaneInfo(
                        lane_count=0,
                        lane_type="unknown",
                        ego_lane_position="unknown",
                        lane_markings=[]
                    ),
                    road_elements=[],
                    surface_condition="unknown"
                ),
                traffic_participants=TrafficParticipants(
                    vehicles=[],
                    cyclists=[],
                    vulnerable_road_users=[]
                ),
                sidewalk_areas=SidewalkArea(
                    has_sidewalk=False,
                    pedestrians=[],
                    static_objects=[],
                    location="unknown"
                ),
                static_infrastructure=StaticInfrastructure(
                    barriers=[],
                    traffic_cones=[],
                    construction=[],
                    other=[]
                ),
                spatial_zones=[],
                safety_critical_elements=["Scene graph generation failed"],
                total_objects=len(annotations)
            )
    
    def _generate_summary(self, scene_graph: HierarchicalSceneGraph) -> str:
        """Generate human-readable summary of scene graph"""
        
        lines = []
        lines.append("=== Hierarchical Scene Graph ===\n")
        
        # Scene summary
        lines.append(f"Scene: {scene_graph.scene_summary}")
        lines.append(f"Total objects: {scene_graph.total_objects}\n")
        
        # Environment
        env = scene_graph.environment
        lines.append(f"Environment:")
        lines.append(f"  - Lighting: {env.lighting}")
        lines.append(f"  - Weather: {env.weather}")
        lines.append(f"  - Location: {env.location_type}\n")
        
        # Road structure
        road = scene_graph.road_structure
        lines.append(f"Road Structure:")
        lines.append(f"  - Type: {road.road_type}")
        lines.append(f"  - Lanes: {road.lanes.lane_count} {road.lanes.lane_type} lanes")
        lines.append(f"  - Ego position: {road.lanes.ego_lane_position} lane")
        if road.road_elements:
            lines.append(f"  - Elements: {len(road.road_elements)} road signs/markings\n")
        
        # Traffic participants
        traffic = scene_graph.traffic_participants
        lines.append(f"Traffic Participants:")
        lines.append(f"  - Vehicles: {len(traffic.vehicles)}")
        lines.append(f"  - Cyclists: {len(traffic.cyclists)}")
        lines.append(f"  - Vulnerable road users: {len(traffic.vulnerable_road_users)}\n")
        
        # Sidewalk
        sidewalk = scene_graph.sidewalk_areas
        if sidewalk.has_sidewalk:
            lines.append(f"Sidewalk Areas ({sidewalk.location}):")
            lines.append(f"  - Pedestrians: {len(sidewalk.pedestrians)}")
            lines.append(f"  - Static objects: {len(sidewalk.static_objects)}\n")
        
        # Static infrastructure
        infra = scene_graph.static_infrastructure
        total_static = (len(infra.barriers) + len(infra.traffic_cones) + 
                       len(infra.construction) + len(infra.other))
        if total_static > 0:
            lines.append(f"Static Infrastructure:")
            if infra.barriers:
                lines.append(f"  - Barriers: {len(infra.barriers)}")
            if infra.traffic_cones:
                lines.append(f"  - Traffic cones: {len(infra.traffic_cones)}")
            if infra.construction:
                lines.append(f"  - Construction: {len(infra.construction)}\n")
        
        # Spatial zones
        if scene_graph.spatial_zones:
            lines.append(f"Spatial Zones:")
            for zone in scene_graph.spatial_zones:
                if zone.objects:
                    lines.append(f"  - {zone.zone_name}: {len(zone.objects)} objects "
                               f"(criticality: {zone.criticality})")
        
        # Safety critical
        if scene_graph.safety_critical_elements:
            lines.append(f"\nSafety Critical Elements:")
            for element in scene_graph.safety_critical_elements:
                lines.append(f"  - {element}")
        
        return '\n'.join(lines)


def create_scenegraph_agent(client, model: str):
    """Factory function to create scene graph agent"""
    return SceneGraphAgent(client, model, "SceneGraphAgent")