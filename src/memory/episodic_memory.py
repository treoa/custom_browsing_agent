"""
Episodic Memory Module

This module provides the EpisodicMemory class which records sequences of research
actions and their outcomes for reflection and learning.
"""

from typing import Dict, List, Any, Optional, Tuple
import time
import json
import uuid
import os

from .base_memory import BaseMemory


class EpisodicMemory(BaseMemory):
    """
    Episodic Memory records sequences of research actions and outcomes.
    
    This memory component stores temporal event sequences, action-result pairings,
    and strategy effectiveness tracking for causal learning.
    """
    
    def __init__(
        self,
        name: str = "Episodic",
        max_episodes: int = 100,
        storage_path: Optional[str] = None,
    ):
        """
        Initialize an EpisodicMemory instance.
        
        Args:
            name: The name/identifier for this memory system (default: "Episodic")
            max_episodes: Maximum number of episodes to store (default: 100)
            storage_path: Path to store the episodes (default: in-memory)
        """
        super().__init__(name=name)
        self.max_episodes = max_episodes
        self.storage_path = storage_path
        
        # Lists to store episodes and events
        self.episodes = []  # List of episode dictionaries
        self.current_episode = None  # Reference to current active episode
        
        # Load existing data if storage path is provided
        if storage_path and os.path.exists(storage_path):
            self._load_from_disk()
    
    async def add(self, key: str, value: Any, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Add an event to the current episode.
        
        Args:
            key: Event type/identifier
            value: Event data
            metadata: Optional metadata about the event
            
        Returns:
            Success status
        """
        # Create metadata if none provided
        metadata = metadata or {}
        
        # Add timestamp if not provided
        if "timestamp" not in metadata:
            metadata["timestamp"] = time.time()
        
        # Add event ID if not provided
        if "event_id" not in metadata:
            metadata["event_id"] = str(uuid.uuid4())
        
        # Create episode if none exists
        if not self.current_episode:
            await self.create_episode(f"episode_{len(self.episodes) + 1}")
        
        # Create event
        event = {
            "event_type": key,
            "data": value,
            "metadata": metadata
        }
        
        # Add to current episode
        self.current_episode["events"].append(event)
        
        # Update episode metadata
        self.current_episode["metadata"]["last_updated"] = time.time()
        self.current_episode["metadata"]["event_count"] = len(self.current_episode["events"])
        
        # Save to disk if storage path is provided
        if self.storage_path:
            self._save_to_disk()
        
        return True
    
    async def get(self, key: str) -> Optional[Any]:
        """
        Retrieve an episode by ID.
        
        Args:
            key: Episode ID
            
        Returns:
            The episode data, or None if not found
        """
        for episode in self.episodes:
            if episode["id"] == key:
                return episode
        
        return None
    
    async def search(self, query: Any, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search episodes for relevant events.
        
        Args:
            query: The search query (string or dict with criteria)
            limit: Maximum number of results to return
            
        Returns:
            List of matching episodes/events with metadata
        """
        results = []
        
        # String-based search
        if isinstance(query, str):
            query_lower = query.lower()
            
            # Search for events containing the query string
            for episode in self.episodes:
                matching_events = []
                
                for event in episode["events"]:
                    # Try to match in event type
                    if query_lower in event["event_type"].lower():
                        matching_events.append(event)
                        continue
                    
                    # Try to match in event data (serialize to string if needed)
                    try:
                        if isinstance(event["data"], str):
                            if query_lower in event["data"].lower():
                                matching_events.append(event)
                                continue
                        else:
                            data_str = json.dumps(event["data"]).lower()
                            if query_lower in data_str:
                                matching_events.append(event)
                                continue
                    except:
                        pass
                    
                    # Try to match in metadata
                    try:
                        metadata_str = json.dumps(event["metadata"]).lower()
                        if query_lower in metadata_str:
                            matching_events.append(event)
                    except:
                        pass
                
                # If episode has matching events, add to results
                if matching_events:
                    results.append({
                        "episode": episode,
                        "matching_events": matching_events
                    })
        
        # Dict-based criteria search
        elif isinstance(query, dict):
            for episode in self.episodes:
                # Match episode-level criteria
                episode_match = True
                for key, value in query.items():
                    if key == "id" and episode["id"] != value:
                        episode_match = False
                        break
                    elif key == "label" and episode["label"] != value:
                        episode_match = False
                        break
                    elif key == "metadata":
                        for metadata_key, metadata_value in value.items():
                            if metadata_key not in episode["metadata"] or episode["metadata"][metadata_key] != metadata_value:
                                episode_match = False
                                break
                
                # If episode matches criteria, add to results
                if episode_match:
                    results.append({
                        "episode": episode,
                        "matching_events": episode["events"]
                    })
                    continue
                
                # If episode doesn't match criteria, check individual events
                matching_events = []
                for event in episode["events"]:
                    event_match = True
                    for key, value in query.items():
                        if key == "event_type" and event["event_type"] != value:
                            event_match = False
                            break
                        elif key == "metadata":
                            for metadata_key, metadata_value in value.items():
                                if metadata_key not in event["metadata"] or event["metadata"][metadata_key] != metadata_value:
                                    event_match = False
                                    break
                    
                    if event_match:
                        matching_events.append(event)
                
                # If any events match criteria, add episode to results
                if matching_events:
                    results.append({
                        "episode": episode,
                        "matching_events": matching_events
                    })
        
        # Sort results by recency (newest first)
        results.sort(key=lambda x: -x["episode"]["metadata"]["timestamp"])
        
        return results[:limit]
    
    async def update(self, key: str, value: Any, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Update an episode's metadata or label.
        
        Args:
            key: Episode ID
            value: New label or None to keep current label
            metadata: New metadata to merge with existing metadata
            
        Returns:
            Success status
        """
        # Find episode
        episode_to_update = None
        for episode in self.episodes:
            if episode["id"] == key:
                episode_to_update = episode
                break
        
        if not episode_to_update:
            return False
        
        # Update label if provided
        if value is not None:
            episode_to_update["label"] = value
        
        # Update metadata if provided
        if metadata:
            episode_to_update["metadata"].update(metadata)
        
        # Update timestamp
        episode_to_update["metadata"]["last_updated"] = time.time()
        
        # Save to disk if storage path is provided
        if self.storage_path:
            self._save_to_disk()
        
        return True
    
    async def delete(self, key: str) -> bool:
        """
        Delete an episode.
        
        Args:
            key: Episode ID
            
        Returns:
            Success status
        """
        # Find episode index
        episode_index = None
        for i, episode in enumerate(self.episodes):
            if episode["id"] == key:
                episode_index = i
                break
        
        if episode_index is None:
            return False
        
        # Check if it's the current episode
        if self.current_episode and self.current_episode["id"] == key:
            self.current_episode = None
        
        # Remove episode
        self.episodes.pop(episode_index)
        
        # Save to disk if storage path is provided
        if self.storage_path:
            self._save_to_disk()
        
        return True
    
    async def clear(self) -> bool:
        """
        Clear all episodes.
        
        Returns:
            Success status
        """
        self.episodes = []
        self.current_episode = None
        
        # Save to disk if storage path is provided
        if self.storage_path:
            self._save_to_disk()
        
        return True
    
    async def create_episode(self, label: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a new episode and set it as current.
        
        Args:
            label: Human-readable label for the episode
            metadata: Optional metadata for the episode
            
        Returns:
            Episode ID
        """
        # Create metadata if none provided
        metadata = metadata or {}
        
        # Add timestamp if not provided
        if "timestamp" not in metadata:
            metadata["timestamp"] = time.time()
        
        # Add last_updated if not provided
        if "last_updated" not in metadata:
            metadata["last_updated"] = metadata["timestamp"]
        
        # Add event_count if not provided
        if "event_count" not in metadata:
            metadata["event_count"] = 0
        
        # Generate ID
        episode_id = str(uuid.uuid4())
        
        # Create episode
        episode = {
            "id": episode_id,
            "label": label,
            "metadata": metadata,
            "events": []
        }
        
        # Add to episodes list
        self.episodes.append(episode)
        
        # Set as current episode
        self.current_episode = episode
        
        # Check if we're over capacity
        if len(self.episodes) > self.max_episodes:
            # Remove oldest episode
            self.episodes.sort(key=lambda x: x["metadata"]["timestamp"])
            self.episodes.pop(0)
        
        # Save to disk if storage path is provided
        if self.storage_path:
            self._save_to_disk()
        
        return episode_id
    
    async def end_episode(self, outcome: Optional[Dict[str, Any]] = None) -> bool:
        """
        End the current episode and mark it with an outcome.
        
        Args:
            outcome: Optional outcome information
            
        Returns:
            Success status
        """
        if not self.current_episode:
            return False
        
        # Add outcome if provided
        if outcome:
            self.current_episode["outcome"] = outcome
        
        # Add end timestamp
        self.current_episode["metadata"]["end_timestamp"] = time.time()
        
        # Calculate duration
        start_time = self.current_episode["metadata"].get("timestamp", 0)
        end_time = self.current_episode["metadata"]["end_timestamp"]
        self.current_episode["metadata"]["duration"] = end_time - start_time
        
        # Clear current episode reference
        self.current_episode = None
        
        # Save to disk if storage path is provided
        if self.storage_path:
            self._save_to_disk()
        
        return True
    
    async def get_current_episode(self) -> Optional[Dict[str, Any]]:
        """
        Get the current active episode.
        
        Returns:
            Current episode or None if no active episode
        """
        return self.current_episode
    
    async def get_event_sequence(self, episode_id: str) -> List[Dict[str, Any]]:
        """
        Get all events in an episode in chronological order.
        
        Args:
            episode_id: Episode ID
            
        Returns:
            List of events
        """
        episode = await self.get(episode_id)
        if not episode:
            return []
        
        # Sort events by timestamp
        events = sorted(episode["events"], key=lambda x: x["metadata"]["timestamp"])
        return events
    
    async def add_action_result_pair(
        self, 
        action: Dict[str, Any], 
        result: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Add an action-result pair to the current episode.
        
        Args:
            action: Action data
            result: Result data
            metadata: Optional metadata
            
        Returns:
            Success status
        """
        # Create metadata if none provided
        metadata = metadata or {}
        
        # Add timestamp if not provided
        if "timestamp" not in metadata:
            metadata["timestamp"] = time.time()
        
        # Add pair ID if not provided
        if "pair_id" not in metadata:
            metadata["pair_id"] = str(uuid.uuid4())
        
        # Create action-result pair
        pair = {
            "action": action,
            "result": result,
            "metadata": metadata
        }
        
        # Add to current episode
        return await self.add("action_result_pair", pair)
    
    async def get_success_patterns(self, min_success_rate: float = 0.7, min_occurrences: int = 3) -> List[Dict[str, Any]]:
        """
        Identify patterns of actions that lead to successful outcomes.
        
        Args:
            min_success_rate: Minimum success rate to consider a pattern (default: 0.7)
            min_occurrences: Minimum number of occurrences to consider a pattern (default: 3)
            
        Returns:
            List of successful patterns with statistics
        """
        # This is a simplified implementation of pattern detection
        # In a production system, would use more sophisticated algorithms
        
        # Collect all action-result pairs
        all_pairs = []
        for episode in self.episodes:
            for event in episode["events"]:
                if event["event_type"] == "action_result_pair":
                    pair = event["data"]
                    # Add episode outcome if available
                    if "outcome" in episode:
                        pair["episode_outcome"] = episode["outcome"]
                    all_pairs.append(pair)
        
        # Group by action type
        action_types = {}
        for pair in all_pairs:
            action_type = pair["action"].get("type", "unknown")
            if action_type not in action_types:
                action_types[action_type] = []
            action_types[action_type].append(pair)
        
        # Analyze success patterns
        success_patterns = []
        for action_type, pairs in action_types.items():
            # Skip if not enough occurrences
            if len(pairs) < min_occurrences:
                continue
            
            # Count successful results
            success_count = 0
            for pair in pairs:
                # Check if result is marked as successful
                if pair["result"].get("success", False):
                    success_count += 1
                # Also check episode outcome if available
                elif "episode_outcome" in pair and pair["episode_outcome"].get("success", False):
                    success_count += 1
            
            # Calculate success rate
            success_rate = success_count / len(pairs)
            
            # Add to patterns if success rate is high enough
            if success_rate >= min_success_rate:
                success_patterns.append({
                    "action_type": action_type,
                    "occurrences": len(pairs),
                    "success_count": success_count,
                    "success_rate": success_rate,
                    "examples": pairs[:3]  # Include a few examples
                })
        
        # Sort by success rate (highest first)
        success_patterns.sort(key=lambda x: -x["success_rate"])
        
        return success_patterns
        
    async def get_recent_items(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get the most recent events across all episodes.
        
        Args:
            limit: Maximum number of events to return
            
        Returns:
            List of recent events with their metadata
        """
        all_events = []
        
        # Collect all events from all episodes
        for episode in self.episodes:
            for event in episode["events"]:
                # Add episode reference
                event_with_context = event.copy()
                event_with_context["episode_id"] = episode["id"]
                event_with_context["episode_label"] = episode["label"]
                all_events.append(event_with_context)
        
        # Sort by timestamp (newest first)
        all_events.sort(key=lambda x: -x["metadata"].get("timestamp", 0))
        
        return all_events[:limit]
    
    async def get_recent_episodes(self, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get the most recently active episodes.
        
        Args:
            limit: Maximum number of episodes to return
            
        Returns:
            List of recent episodes
        """
        # Sort episodes by last_updated timestamp (newest first)
        recent_episodes = sorted(
            self.episodes,
            key=lambda x: -x["metadata"].get("last_updated", x["metadata"].get("timestamp", 0))
        )
        
        return recent_episodes[:limit]
    
    async def get_episode_by_label(self, label: str) -> Optional[Dict[str, Any]]:
        """
        Find an episode by its label.
        
        Args:
            label: Episode label to search for
            
        Returns:
            Matching episode or None if not found
        """
        for episode in self.episodes:
            if episode["label"] == label:
                return episode
        
        return None
    
    async def extract_insights(self, episode_id: str) -> Dict[str, Any]:
        """
        Extract insights and patterns from a specific episode.
        
        Args:
            episode_id: Episode to analyze
            
        Returns:
            Dictionary of insights
        """
        episode = await self.get(episode_id)
        if not episode:
            return {"error": "Episode not found"}
        
        insights = {
            "episode_id": episode_id,
            "episode_label": episode["label"],
            "duration": episode["metadata"].get("duration", 0),
            "success": False,
            "event_count": len(episode["events"]),
            "key_actions": [],
            "bottlenecks": [],
            "patterns": []
        }
        
        # Check success status from outcome
        if "outcome" in episode:
            insights["success"] = episode["outcome"].get("success", False)
            insights["outcome_details"] = episode["outcome"]
        
        # Identify key actions (actions that led to significant events or outcomes)
        action_result_pairs = []
        for event in episode["events"]:
            if event["event_type"] == "action_result_pair":
                action_result_pairs.append(event["data"])
        
        # Identify successful actions
        successful_actions = []
        for pair in action_result_pairs:
            if pair["result"].get("success", False):
                successful_actions.append(pair)
        
        # Identify bottlenecks (failed actions or slow actions)
        bottlenecks = []
        for pair in action_result_pairs:
            if pair["result"].get("error", None) or not pair["result"].get("success", True):
                bottlenecks.append(pair)
        
        insights["key_actions"] = successful_actions[:5]  # Include top 5 successful actions
        insights["bottlenecks"] = bottlenecks[:5]  # Include top 5 bottlenecks
        
        return insights
    
    def _save_to_disk(self) -> None:
        """
        Save episodes to disk.
        """
        if not self.storage_path:
            return
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        
        # Save data
        with open(self.storage_path, 'w') as f:
            json.dump({"episodes": self.episodes}, f)
    
    def _load_from_disk(self) -> None:
        """
        Load episodes from disk.
        """
        if not self.storage_path or not os.path.exists(self.storage_path):
            return
        
        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
                self.episodes = data.get("episodes", [])
        except Exception as e:
            print(f"Error loading episodic memory: {e}")
            # Initialize empty episodes list
            self.episodes = []