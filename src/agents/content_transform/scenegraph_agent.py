"""
Scene Graph Agent with Hierarchical Structure
Builds rich, hierarchical scene graphs from object annotations
"""

from typing import Any, Dict, Optional, List, Tuple, Set
from agents import BaseAgent
import numpy as np
import json
from dataclasses import dataclass, asdict
from collections import defaultdict
import networkx as nx


@dataclass
class SceneObject:
    """Represents a single object in the scene"""
    id: str
    category: str
    subcategory: str  # e.g., "car" -> "sedan", "suv"
    position: np.ndarray  # [x, y, z]
    velocity: np.ndarray  # [vx, vy, vz]
    dimensions: np.ndarray  # [width, length, height]
    attributes: List[str]
    visibility: float  # 0-1
    distance_to_ego: float
    angle_to_ego: float  # degrees
    region: str  # 'front', 'back', 'left', 'right', etc.
    
    def to_dict(self) -> Dict:
        """Convert to serializable dict"""
        return {
            'id': self.id,
            'category': self.category,
            'subcategory': self.subcategory,
            'position': self.position.tolist(),
            'velocity': self.velocity.tolist(),
            'dimensions': self.dimensions.tolist(),
            'attributes': self.attributes,
            'visibility': float(self.visibility),
            'distance_to_ego': float(self.distance_to_ego),
            'angle_to_ego': float(self.angle_to_ego),
            'region': self.region
        }


@dataclass
class SpatialRelation:
    """Represents a spatial relationship between objects"""
    subject_id: str
    relation_type: str  # 'in_front_of', 'behind', 'left_of', 'right_of', 'near', 'far_from'
    object_id: str
    distance: float
    confidence: float


@dataclass
class ObjectGroup:
    """Represents a group of related objects"""
    group_id: str
    group_type: str  # 'vehicle_cluster', 'pedestrian_group', 'traffic_infrastructure'
    object_ids: List[str]
    center_position: np.ndarray
    spatial_extent: float
    description: str


class SceneGraphAgent(BaseAgent):
    """
    Builds hierarchical scene graphs with:
    1. Scene-level context (traffic density, environment type)
    2. Object groups (clusters of related objects)
    3. Individual objects with rich attributes
    4. Spatial relationships between objects
    5. Temporal information (moving vs static)
    """
    
    def __init__(self, client, model: str, agent_name: str):
        super().__init__(client, model, agent_name)
        
        # Spatial relationship thresholds
        self.near_threshold = 5.0  # meters
        self.medium_threshold = 15.0
        self.far_threshold = 30.0
        
        # Category hierarchy (parent -> children mapping)
        self.category_hierarchy = {
            'vehicle': ['car', 'truck', 'bus', 'construction_vehicle', 'trailer'],
            'vulnerable_road_user': ['pedestrian', 'bicycle', 'motorcycle'],
            'static_object': ['barrier', 'traffic_cone', 'pushable_pullable'],
            'background': ['driveable_surface', 'sidewalk', 'terrain']
        }
        
        # Inverse mapping (child -> parent)
        self.parent_categories = {}
        for parent, children in self.category_hierarchy.items():
            for child in children:
                self.parent_categories[child] = parent
    
    def process(self, annotations: List[Dict], context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Process object annotations into hierarchical scene graph
        
        Args:
            annotations: List of nuScenes object annotations
            context: Optional context from other agents
        """        
        # Step 1: Parse and enrich annotations
        scene_objects = self._parse_annotations(annotations)
        
        # Step 2: Compute spatial relationships
        spatial_relations = self._compute_spatial_relations(scene_objects)
        
        # Step 3: Group objects by proximity and type
        object_groups = self._create_object_groups(scene_objects, spatial_relations)
        
        # Step 4: Build hierarchical structure
        hierarchy = self._build_hierarchy(scene_objects, object_groups, spatial_relations)
        
        # Step 5: Extract scene-level features
        scene_context = self._extract_scene_context(scene_objects, object_groups)
        
        # Step 6: Build networkx graph for visualization and queries
        graph = self._build_graph(scene_objects, spatial_relations, object_groups)
        
        # Step 7: Generate structured report
        structured_report = self._generate_structured_report(
            hierarchy, scene_context, scene_objects, object_groups
        )
        
        # Step 8: Use LLM for high-level semantic interpretation
        semantic_interpretation = self._generate_semantic_interpretation(
            structured_report, hierarchy, scene_context, context
        )
        
        return {
            "agent": self.agent_name,
            "modality": "scene_graph",
            "hierarchy": hierarchy,
            "scene_context": scene_context,
            "objects": [obj.to_dict() for obj in scene_objects],
            "spatial_relations": [asdict(rel) for rel in spatial_relations],
            "object_groups": [self._group_to_dict(grp) for grp in object_groups],
            "graph_stats": {
                "num_nodes": graph.number_of_nodes(),
                "num_edges": graph.number_of_edges(),
                "density": nx.density(graph)
            },
            "structured_report": structured_report,
            "observations": semantic_interpretation
        }
    
    def _parse_annotations(self, annotations: List[Dict]) -> List[SceneObject]:
        """Parse nuScenes annotations into rich SceneObject instances"""
        scene_objects = []
        
        for i, ann in enumerate(annotations):
            # Extract basic info
            category = self._normalize_category(ann.get('category_name', 'unknown'))
            position = np.array(ann.get('translation', [0, 0, 0]))
            
            # Velocity (might be None)
            velocity_raw = ann.get('velocity', [0, 0])
            if velocity_raw is None or any(v is None for v in velocity_raw):
                velocity = np.array([0.0, 0.0, 0.0])
            else:
                velocity = np.array([velocity_raw[0], velocity_raw[1], 0.0])
            
            dimensions = np.array(ann.get('size', [0, 0, 0]))
            
            # Parse visibility
            visibility_token = ann.get('visibility_token', 'unknown')
            visibility = self._parse_visibility(visibility_token)
            
            # Extract attributes
            attributes = []
            for attr in ann.get('attribute_tokens', []):
                if isinstance(attr, str):
                    attributes.append(attr)
            
            # Determine if moving
            speed = np.linalg.norm(velocity)
            if speed > 0.5:  # m/s
                if 'vehicle.stopped' not in attributes and 'vehicle.parked' not in attributes:
                    attributes.append('moving')
            elif speed < 0.1:
                if 'vehicle' in category:
                    attributes.append('stationary')
            
            # Compute ego-relative position
            distance_to_ego = np.linalg.norm(position[:2])  # XY distance only
            angle_to_ego = np.arctan2(position[1], position[0]) * 180 / np.pi
            region = self._get_region(position[:2])
            
            # Determine subcategory
            subcategory = self._get_subcategory(category, dimensions, attributes)
            
            scene_objects.append(SceneObject(
                id=f"obj_{i}",
                category=category,
                subcategory=subcategory,
                position=position,
                velocity=velocity,
                dimensions=dimensions,
                attributes=attributes,
                visibility=visibility,
                distance_to_ego=distance_to_ego,
                angle_to_ego=angle_to_ego,
                region=region
            ))
        
        return scene_objects
    
    def _normalize_category(self, category: str) -> str:
        """Normalize nuScenes category names"""
        category = category.lower()
        
        # Remove prefixes
        prefixes = ['vehicle.', 'human.pedestrian.', 'movable_object.', 
                   'static_object.', 'animal.']
        for prefix in prefixes:
            if category.startswith(prefix):
                category = category[len(prefix):]
        
        # Map to standard categories
        mapping = {
            'adult': 'pedestrian',
            'child': 'pedestrian',
            'construction_worker': 'pedestrian',
            'police_officer': 'pedestrian',
            'rigid': 'bus',
            'bendy': 'bus',
            'pushable_pullable': 'pushable_object'
        }
        
        return mapping.get(category, category)
    
    def _parse_visibility(self, visibility_token: str) -> float:
        """Parse visibility token to 0-1 scale"""
        if isinstance(visibility_token, (int, float)):
            return float(visibility_token)
        
        # Extract percentage from string like "visibility_60_80" or "60-80%"
        import re
        numbers = re.findall(r'\d+', str(visibility_token))
        
        if len(numbers) >= 2:
            # Take average of range
            return (int(numbers[0]) + int(numbers[1])) / 200.0
        elif len(numbers) == 1:
            return int(numbers[0]) / 100.0
        
        return 0.5  # Default
    
    def _get_region(self, position_2d: np.ndarray) -> str:
        """Get detailed region label (8 directions + distance)"""
        x, y = position_2d
        distance = np.sqrt(x**2 + y**2)
        angle = np.arctan2(y, x) * 180 / np.pi
        angle = (angle + 360) % 360
        
        # Distance qualifier
        if distance < 10:
            dist_qual = "close"
        elif distance < 30:
            dist_qual = "medium"
        else:
            dist_qual = "far"
        
        # Direction (8 directions)
        if 337.5 <= angle or angle < 22.5:
            direction = "front_right"
        elif 22.5 <= angle < 67.5:
            direction = "front"
        elif 67.5 <= angle < 112.5:
            direction = "front_left"
        elif 112.5 <= angle < 157.5:
            direction = "left"
        elif 157.5 <= angle < 202.5:
            direction = "back_left"
        elif 202.5 <= angle < 247.5:
            direction = "back"
        elif 247.5 <= angle < 292.5:
            direction = "back_right"
        else:
            direction = "right"
        
        return f"{dist_qual}_{direction}"
    
    def _get_subcategory(self, category: str, dimensions: np.ndarray, 
                        attributes: List[str]) -> str:
        """Infer subcategory from dimensions and attributes"""
        length, width, height = dimensions
        
        if category == 'car':
            if length > 5.0:
                return 'large_car'
            elif length < 4.0:
                return 'compact_car'
            else:
                return 'sedan'
        
        elif category == 'truck':
            if length > 8.0:
                return 'semi_truck'
            elif length > 6.0:
                return 'large_truck'
            else:
                return 'pickup_truck'
        
        elif category == 'pedestrian':
            if 'construction_worker' in str(attributes):
                return 'construction_worker'
            elif 'police_officer' in str(attributes):
                return 'police_officer'
            elif height < 1.3:
                return 'child'
            else:
                return 'adult'
        
        elif category == 'bicycle':
            if any('motorcycle' in str(attr) for attr in attributes):
                return 'motorcycle'
            else:
                return 'bicycle'
        
        return category
    
    def _compute_spatial_relations(self, objects: List[SceneObject]) -> List[SpatialRelation]:
        """Compute spatial relationships between all object pairs"""
        relations = []
        
        for i, obj1 in enumerate(objects):
            for j, obj2 in enumerate(objects):
                if i >= j:  # Avoid duplicates and self-relations
                    continue
                
                # Compute distance
                distance = np.linalg.norm(obj1.position - obj2.position)
                
                # Determine relation type
                relation_types = []
                
                # Distance-based relations
                if distance < self.near_threshold:
                    relation_types.append(('near', 1.0))
                elif distance < self.medium_threshold:
                    relation_types.append(('medium_distance', 0.8))
                elif distance < self.far_threshold:
                    relation_types.append(('far_from', 0.6))
                
                # Directional relations (relative to obj1)
                relative_pos = obj2.position - obj1.position
                
                # Front/Back (Y axis)
                if relative_pos[1] > 2.0:
                    relation_types.append(('in_front_of', 0.9))
                elif relative_pos[1] < -2.0:
                    relation_types.append(('behind', 0.9))
                
                # Left/Right (X axis)
                if relative_pos[0] > 2.0:
                    relation_types.append(('right_of', 0.9))
                elif relative_pos[0] < -2.0:
                    relation_types.append(('left_of', 0.9))
                
                # Add all relations
                for rel_type, confidence in relation_types:
                    relations.append(SpatialRelation(
                        subject_id=obj1.id,
                        relation_type=rel_type,
                        object_id=obj2.id,
                        distance=distance,
                        confidence=confidence
                    ))
        
        return relations
    
    def _create_object_groups(self, objects: List[SceneObject],
                             relations: List[SpatialRelation]) -> List[ObjectGroup]:
        """Group nearby objects of similar types"""
        groups = []
        grouped_ids = set()
        
        # Group by category and proximity
        category_objects = defaultdict(list)
        for obj in objects:
            parent_cat = self.parent_categories.get(obj.category, 'other')
            category_objects[parent_cat].append(obj)
        
        for parent_cat, cat_objects in category_objects.items():
            if len(cat_objects) < 2:
                continue
            
            # Find clusters of nearby objects
            positions = np.array([obj.position for obj in cat_objects])
            
            # Simple distance-based clustering
            visited = set()
            
            for i, obj in enumerate(cat_objects):
                if obj.id in visited or obj.id in grouped_ids:
                    continue
                
                # Find nearby objects of same category
                cluster = [obj]
                cluster_ids = {obj.id}
                
                for j, other_obj in enumerate(cat_objects):
                    if i == j or other_obj.id in visited:
                        continue
                    
                    dist = np.linalg.norm(obj.position - other_obj.position)
                    if dist < 10.0:  # 10 meter clustering threshold
                        cluster.append(other_obj)
                        cluster_ids.add(other_obj.id)
                        visited.add(other_obj.id)
                
                if len(cluster) >= 2:
                    # Create group
                    center = np.mean([o.position for o in cluster], axis=0)
                    extent = max(np.linalg.norm(o.position - center) for o in cluster)
                    
                    description = f"Group of {len(cluster)} {parent_cat}s"
                    if parent_cat == 'vehicle':
                        description = f"Vehicle cluster ({len(cluster)} vehicles)"
                    elif parent_cat == 'vulnerable_road_user':
                        description = f"Pedestrian/cyclist group ({len(cluster)} individuals)"
                    
                    groups.append(ObjectGroup(
                        group_id=f"group_{len(groups)}",
                        group_type=f"{parent_cat}_cluster",
                        object_ids=list(cluster_ids),
                        center_position=center,
                        spatial_extent=extent,
                        description=description
                    ))
                    
                    grouped_ids.update(cluster_ids)
                    visited.add(obj.id)
        
        return groups
    
    def _build_hierarchy(self, objects: List[SceneObject],
                        groups: List[ObjectGroup],
                        relations: List[SpatialRelation]) -> Dict:
        """Build hierarchical scene representation"""
        
        hierarchy = {
            'scene': {
                'total_objects': len(objects),
                'regions': {},
                'categories': {},
                'groups': {}
            }
        }
        
        # Group by region
        for obj in objects:
            region = obj.region
            if region not in hierarchy['scene']['regions']:
                hierarchy['scene']['regions'][region] = {
                    'object_ids': [],
                    'count': 0,
                    'categories': defaultdict(int)
                }
            
            hierarchy['scene']['regions'][region]['object_ids'].append(obj.id)
            hierarchy['scene']['regions'][region]['count'] += 1
            hierarchy['scene']['regions'][region]['categories'][obj.category] += 1
        
        # Convert defaultdict to regular dict
        for region_data in hierarchy['scene']['regions'].values():
            region_data['categories'] = dict(region_data['categories'])
        
        # Group by parent category
        for obj in objects:
            parent_cat = self.parent_categories.get(obj.category, 'other')
            
            if parent_cat not in hierarchy['scene']['categories']:
                hierarchy['scene']['categories'][parent_cat] = {
                    'subcategories': {},
                    'count': 0,
                    'object_ids': []
                }
            
            hierarchy['scene']['categories'][parent_cat]['count'] += 1
            hierarchy['scene']['categories'][parent_cat]['object_ids'].append(obj.id)
            
            # Subcategory
            if obj.category not in hierarchy['scene']['categories'][parent_cat]['subcategories']:
                hierarchy['scene']['categories'][parent_cat]['subcategories'][obj.category] = {
                    'instances': [],
                    'count': 0
                }
            
            hierarchy['scene']['categories'][parent_cat]['subcategories'][obj.category]['count'] += 1
            hierarchy['scene']['categories'][parent_cat]['subcategories'][obj.category]['instances'].append({
                'id': obj.id,
                'subcategory': obj.subcategory,
                'position': obj.position.tolist(),
                'region': obj.region,
                'attributes': obj.attributes
            })
        
        # Add groups
        for group in groups:
            hierarchy['scene']['groups'][group.group_id] = {
                'type': group.group_type,
                'object_ids': group.object_ids,
                'description': group.description,
                'center': group.center_position.tolist(),
                'extent': float(group.spatial_extent)
            }
        
        return hierarchy
    
    def _extract_scene_context(self, objects: List[SceneObject],
                               groups: List[ObjectGroup]) -> Dict[str, Any]:
        """Extract high-level scene context features"""
        
        # Count categories
        category_counts = defaultdict(int)
        parent_counts = defaultdict(int)
        
        for obj in objects:
            category_counts[obj.category] += 1
            parent = self.parent_categories.get(obj.category, 'other')
            parent_counts[parent] += 1
        
        # Analyze movement
        moving_objects = [obj for obj in objects if 'moving' in obj.attributes]
        stationary_objects = [obj for obj in objects if 'stationary' in obj.attributes]
        
        # Traffic density estimation
        vehicle_count = parent_counts.get('vehicle', 0)
        if vehicle_count > 15:
            traffic_density = 'heavy'
        elif vehicle_count > 8:
            traffic_density = 'moderate'
        elif vehicle_count > 3:
            traffic_density = 'light'
        else:
            traffic_density = 'sparse'
        
        # Scene type estimation
        scene_type = 'unknown'
        if parent_counts.get('vulnerable_road_user', 0) > 5:
            scene_type = 'urban_intersection'
        elif vehicle_count > 10:
            scene_type = 'highway'
        elif parent_counts.get('static_object', 0) > 5:
            scene_type = 'construction_zone'
        else:
            scene_type = 'urban_road'
        
        # Spatial distribution
        front_objects = [obj for obj in objects if 'front' in obj.region]
        close_objects = [obj for obj in objects if 'close' in obj.region]
        
        return {
            'total_objects': len(objects),
            'category_counts': dict(category_counts),
            'parent_category_counts': dict(parent_counts),
            'traffic_density': traffic_density,
            'scene_type': scene_type,
            'movement_summary': {
                'moving': len(moving_objects),
                'stationary': len(stationary_objects),
                'moving_percentage': len(moving_objects) / len(objects) if objects else 0
            },
            'spatial_summary': {
                'front_objects': len(front_objects),
                'close_objects': len(close_objects),
                'num_groups': len(groups)
            },
            'visibility_stats': {
                'high_visibility': len([o for o in objects if o.visibility > 0.8]),
                'medium_visibility': len([o for o in objects if 0.4 < o.visibility <= 0.8]),
                'low_visibility': len([o for o in objects if o.visibility <= 0.4])
            }
        }
    
    def _build_graph(self, objects: List[SceneObject],
                    relations: List[SpatialRelation],
                    groups: List[ObjectGroup]) -> nx.Graph:
        """Build networkx graph for advanced queries"""
        G = nx.Graph()
        
        # Add object nodes
        for obj in objects:
            G.add_node(obj.id, 
                      category=obj.category,
                      position=obj.position,
                      region=obj.region)
        
        # Add relation edges
        for rel in relations:
            if rel.confidence > 0.7:  # Only strong relations
                G.add_edge(rel.subject_id, rel.object_id,
                          relation=rel.relation_type,
                          distance=rel.distance)
        
        return G
    
    def _generate_structured_report(self, hierarchy: Dict,
                                   scene_context: Dict,
                                   objects: List[SceneObject],
                                   groups: List[ObjectGroup]) -> str:
        """Generate human-readable structured report"""
        lines = []
        
        lines.append("=== Scene Graph Analysis ===\n")
        
        # Scene context
        lines.append(f"Scene Type: {scene_context['scene_type']}")
        lines.append(f"Traffic Density: {scene_context['traffic_density']}")
        lines.append(f"Total Objects: {scene_context['total_objects']}")
        
        # Category breakdown
        lines.append("\nObject Categories:")
        for parent, count in scene_context['parent_category_counts'].items():
            lines.append(f"  - {parent}: {count}")
            
            # Show subcategories
            if parent in hierarchy['scene']['categories']:
                subcat_data = hierarchy['scene']['categories'][parent]['subcategories']
                for subcat, subcat_info in subcat_data.items():
                    lines.append(f"    • {subcat}: {subcat_info['count']}")
        
        # Spatial distribution
        lines.append("\nSpatial Distribution by Region:")
        for region, region_data in sorted(hierarchy['scene']['regions'].items()):
            if region_data['count'] > 0:
                categories_str = ', '.join(f"{cat}({cnt})" 
                                          for cat, cnt in region_data['categories'].items())
                lines.append(f"  - {region}: {region_data['count']} objects ({categories_str})")
        
        # Movement analysis
        movement = scene_context['movement_summary']
        lines.append(f"\nMovement Analysis:")
        lines.append(f"  - Moving objects: {movement['moving']}")
        lines.append(f"  - Stationary objects: {movement['stationary']}")
        lines.append(f"  - Movement ratio: {movement['moving_percentage']:.1%}")
        
        # Object groups
        if groups:
            lines.append(f"\nObject Groups ({len(groups)} clusters):")
            for group in groups:
                lines.append(f"  - {group.description}")
                lines.append(f"    Members: {len(group.object_ids)} objects")
                lines.append(f"    Spatial extent: {group.spatial_extent:.1f}m")
        
        # Critical objects (close, moving)
        close_moving = [obj for obj in objects 
                       if 'close' in obj.region and 'moving' in obj.attributes]
        if close_moving:
            lines.append(f"\nCritical Objects (close & moving): {len(close_moving)}")
            for obj in close_moving[:5]:  # Top 5
                lines.append(f"  - {obj.category} at {obj.distance_to_ego:.1f}m ({obj.region})")
        
        return '\n'.join(lines)
    
    def _generate_semantic_interpretation(self, structured_report: str,
                                         hierarchy: Dict,
                                         scene_context: Dict,
                                         context: Optional[Dict]) -> str:
        """Use LLM for high-level semantic interpretation"""
        
        system_prompt = """You are an autonomous driving scene understanding expert specializing in scene graph interpretation.

You receive a hierarchical scene graph with:
- Scene-level context (traffic density, scene type)
- Object category hierarchy (vehicles, pedestrians, static objects)
- Spatial distribution by region
- Object groupings and clusters
- Movement analysis

Your task:
- Provide high-level semantic interpretation of the scene
- Identify key safety-relevant patterns
- Describe the overall driving context
- Note any unusual or important configurations
- Explain relationships between object groups

Be concise and focus on actionable driving insights."""

        user_prompt = f"""Analyze this hierarchical scene graph:

{structured_report}

Additional Context:
- Scene type: {scene_context['scene_type']}
- Traffic density: {scene_context['traffic_density']}
- Object groups: {len(hierarchy['scene']['groups'])}

Provide a high-level semantic interpretation focusing on:
1. Overall scene understanding
2. Key spatial patterns
3. Safety-critical elements
4. Object group interactions
5. Notable scene characteristics"""

        if context:
            user_prompt += f"\n\nContext from other sensors:\n{json.dumps(context, indent=2)}"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        return self.call_llm(messages, temperature=0.4)
    
    @staticmethod
    def _group_to_dict(group: ObjectGroup) -> Dict:
        """Convert ObjectGroup to dict"""
        return {
            'group_id': group.group_id,
            'group_type': group.group_type,
            'object_ids': group.object_ids,
            'center_position': group.center_position.tolist(),
            'spatial_extent': float(group.spatial_extent),
            'description': group.description
        }


def create_scenegraph_agent(client, model: str):
    """Factory function to create scene graph agent"""
    return SceneGraphAgent(client, model, "SceneGraphAgent")