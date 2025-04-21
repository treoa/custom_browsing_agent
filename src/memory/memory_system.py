"""
Memory System Module

This module provides the MemorySystem class which integrates different memory types
to create a cohesive memory architecture for the autonomous research agent.
"""

from typing import Dict, List, Any, Optional, Union, Type
from langchain_core.language_models import BaseChatModel

from .base_memory import BaseMemory
from .short_term_memory import ShortTermMemory
from .working_memory import WorkingMemory
from .long_term_memory import LongTermMemory
from .episodic_memory import EpisodicMemory


class MemorySystem:
    """
    Memory System integrates different memory types into a cohesive architecture.
    
    This class coordinates across memory types for operations including cross-memory
    indexing, attention-based retrieval, consistency maintenance, and conflict resolution.
    """
    
    def __init__(
        self,
        llm: BaseChatModel,
        stm_capacity: int = 100,
        working_capacity: int = 7,
        ltm_storage_path: Optional[str] = None,
        episodic_storage_path: Optional[str] = None,
    ):
        """
        Initialize a MemorySystem instance.
        
        Args:
            llm: The language model to use
            stm_capacity: Capacity for Short-Term Memory (default: 100)
            working_capacity: Capacity for Working Memory (default: 7)
            ltm_storage_path: Storage path for Long-Term Memory (default: in-memory)
            episodic_storage_path: Storage path for Episodic Memory (default: in-memory)
        """
        self.llm = llm
        
        # Initialize memory components
        self.stm = ShortTermMemory(capacity=stm_capacity)
        self.working_memory = WorkingMemory(capacity=working_capacity)
        self.ltm = LongTermMemory(llm=llm, storage_path=ltm_storage_path)
        self.episodic = EpisodicMemory(storage_path=episodic_storage_path)
        
        # Memory component dictionary for easier access
        self.memories = {
            "stm": self.stm,
            "working": self.working_memory,
            "ltm": self.ltm,
            "episodic": self.episodic
        }
    
    async def add(
        self,
        key: str,
        value: Any,
        memory_types: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, bool]:
        """
        Add an item to specified memory types.
        
        Args:
            key: The key to store the value under
            value: The value to store
            memory_types: List of memory types to add to (default: ["stm"])
            metadata: Optional metadata about the value
            
        Returns:
            Dictionary of memory types and their add success status
        """
        memory_types = memory_types or ["stm"]
        results = {}
        
        for memory_type in memory_types:
            if memory_type in self.memories:
                success = await self.memories[memory_type].add(key, value, metadata)
                results[memory_type] = success
        
        # Implement automatic memory transfer based on priority
        if "stm" in memory_types and metadata and metadata.get("priority", 0) >= 8:
            # High priority items automatically go to working memory
            if "working" not in memory_types:
                success = await self.working_memory.add(key, value, metadata)
                results["working"] = success
        
        return results
    
    async def get(
        self,
        key: str,
        memory_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Retrieve an item from specified memory types.
        
        Args:
            key: The key to retrieve
            memory_types: List of memory types to retrieve from (default: all)
            
        Returns:
            Dictionary of memory types and their retrieved values
        """
        memory_types = memory_types or list(self.memories.keys())
        results = {}
        
        for memory_type in memory_types:
            if memory_type in self.memories:
                value = await self.memories[memory_type].get(key)
                if value is not None:
                    results[memory_type] = value
        
        return results
    
    async def search(
        self,
        query: Any,
        memory_types: Optional[List[str]] = None,
        limit: int = 5
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Search across specified memory types.
        
        Args:
            query: The search query
            memory_types: List of memory types to search (default: all)
            limit: Maximum results per memory type
            
        Returns:
            Dictionary of memory types and their search results
        """
        memory_types = memory_types or list(self.memories.keys())
        results = {}
        
        for memory_type in memory_types:
            if memory_type in self.memories:
                search_results = await self.memories[memory_type].search(query, limit)
                if search_results:
                    results[memory_type] = search_results
        
        return results
    
    async def update(
        self,
        key: str,
        value: Any,
        memory_types: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, bool]:
        """
        Update an item in specified memory types.
        
        Args:
            key: The key to update
            value: The new value
            memory_types: List of memory types to update (default: all where key exists)
            metadata: Optional new metadata
            
        Returns:
            Dictionary of memory types and their update success status
        """
        # If no memory types specified, check where the key exists
        if not memory_types:
            memory_types = []
            for memory_type, memory in self.memories.items():
                existing_value = await memory.get(key)
                if existing_value is not None:
                    memory_types.append(memory_type)
        
        results = {}
        
        for memory_type in memory_types:
            if memory_type in self.memories:
                success = await self.memories[memory_type].update(key, value, metadata)
                results[memory_type] = success
        
        return results
    
    async def delete(
        self,
        key: str,
        memory_types: Optional[List[str]] = None
    ) -> Dict[str, bool]:
        """
        Delete an item from specified memory types.
        
        Args:
            key: The key to delete
            memory_types: List of memory types to delete from (default: all)
            
        Returns:
            Dictionary of memory types and their delete success status
        """
        memory_types = memory_types or list(self.memories.keys())
        results = {}
        
        for memory_type in memory_types:
            if memory_type in self.memories:
                success = await self.memories[memory_type].delete(key)
                results[memory_type] = success
        
        return results
    
    async def clear(
        self,
        memory_types: Optional[List[str]] = None
    ) -> Dict[str, bool]:
        """
        Clear specified memory types.
        
        Args:
            memory_types: List of memory types to clear (default: all)
            
        Returns:
            Dictionary of memory types and their clear success status
        """
        memory_types = memory_types or list(self.memories.keys())
        results = {}
        
        for memory_type in memory_types:
            if memory_type in self.memories:
                success = await self.memories[memory_type].clear()
                results[memory_type] = success
        
        return results
    
    async def transfer(
        self,
        key: str,
        source_memory: str,
        target_memory: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Transfer an item from one memory type to another.
        
        Args:
            key: The key to transfer
            source_memory: Source memory type
            target_memory: Target memory type
            metadata: Optional metadata to add during transfer
            
        Returns:
            Success status
        """
        if source_memory not in self.memories or target_memory not in self.memories:
            return False
        
        # Get the item from source memory
        value = await self.memories[source_memory].get(key)
        if value is None:
            return False
        
        # Add to target memory
        success = await self.memories[target_memory].add(key, value, metadata)
        
        return success
    
    async def consolidate_stm_to_ltm(self, min_priority: int = 7) -> Dict[str, Any]:
        """
        Consolidate important items from STM to LTM.
        
        Args:
            min_priority: Minimum priority to transfer (default: 7)
            
        Returns:
            Dictionary with consolidation statistics
        """
        # Search for high-priority items in STM
        query = {"metadata": {"priority": {"$gte": min_priority}}}
        high_priority_items = await self.stm.search(query, limit=100)
        
        if not high_priority_items:
            return {"transferred": 0, "total": 0}
        
        # Transfer to LTM
        transferred = 0
        for item in high_priority_items:
            key = item["key"]
            value = item["value"]
            metadata = item["metadata"]
            
            # Add to LTM
            success = await self.ltm.add(key, value, metadata)
            if success:
                transferred += 1
        
        return {
            "transferred": transferred,
            "total": len(high_priority_items)
        }
    
    async def retrieve_context(
        self,
        query: Any,
        context_size: int = 10
    ) -> Dict[str, Any]:
        """
        Retrieve a comprehensive context based on a query.
        
        Searches across all memory types to build a rich context.
        
        Args:
            query: The query to search for
            context_size: Maximum context items per memory type
            
        Returns:
            Comprehensive context from different memory sources
        """
        context = {
            "working_memory": [],
            "stm": [],
            "ltm": [],
            "episodic": []
        }
        
        # First check working memory (currently in focus)
        working_focus = await self.working_memory.get_focus()
        if working_focus:
            context["working_memory"] = working_focus
        
        # Search STM
        stm_results = await self.stm.search(query, limit=context_size)
        if stm_results:
            context["stm"] = stm_results
        
        # Search LTM
        ltm_results = await self.ltm.search(query, limit=context_size)
        if ltm_results:
            context["ltm"] = ltm_results
        
        # Search Episodic Memory
        episodic_results = await self.episodic.search(query, limit=context_size)
        if episodic_results:
            context["episodic"] = episodic_results
        
        return context
    
    async def detect_conflicts(
        self,
        key: str
    ) -> List[Dict[str, Any]]:
        """
        Detect conflicts for an item across memory types.
        
        Args:
            key: The key to check for conflicts
            
        Returns:
            List of detected conflicts
        """
        conflicts = []
        values = {}
        
        # Get values from all memory types
        for memory_type, memory in self.memories.items():
            value = await memory.get(key)
            if value is not None:
                values[memory_type] = value
        
        # If item exists in only one memory type, no conflicts
        if len(values) <= 1:
            return conflicts
        
        # Compare values to detect conflicts
        memory_types = list(values.keys())
        for i in range(len(memory_types)):
            for j in range(i + 1, len(memory_types)):
                type_a = memory_types[i]
                type_b = memory_types[j]
                
                # Simple equality check (could be more sophisticated)
                if values[type_a] != values[type_b]:
                    conflicts.append({
                        "key": key,
                        "memory_types": [type_a, type_b],
                        "values": [values[type_a], values[type_b]]
                    })
        
        return conflicts
    
    async def resolve_conflict(
        self,
        conflict: Dict[str, Any],
        resolution_type: str = "newest"
    ) -> bool:
        """
        Resolve a detected conflict.
        
        Args:
            conflict: Conflict information
            resolution_type: How to resolve ("newest", "highest_priority", "ltm_preferred")
            
        Returns:
            Success status
        """
        key = conflict["key"]
        memory_types = conflict["memory_types"]
        values = conflict["values"]
        
        # Get metadata for each value to help with resolution
        metadata = {}
        for i, memory_type in enumerate(memory_types):
            if memory_type in self.memories:
                item = await self.memories[memory_type].get(key)
                if item is not None and hasattr(item, "metadata"):
                    metadata[memory_type] = item.metadata
        
        # Determine which value to keep based on resolution type
        resolved_value = None
        resolved_metadata = None
        preferred_memory = None
        
        if resolution_type == "newest":
            # Use the newest value based on timestamp
            newest_time = 0
            for memory_type in memory_types:
                if memory_type in metadata:
                    timestamp = metadata[memory_type].get("timestamp", 0)
                    if timestamp > newest_time:
                        newest_time = timestamp
                        preferred_memory = memory_type
                        resolved_value = values[memory_types.index(memory_type)]
                        resolved_metadata = metadata[memory_type]
        
        elif resolution_type == "highest_priority":
            # Use value with highest priority
            highest_priority = -1
            for memory_type in memory_types:
                if memory_type in metadata:
                    priority = metadata[memory_type].get("priority", 0)
                    if priority > highest_priority:
                        highest_priority = priority
                        preferred_memory = memory_type
                        resolved_value = values[memory_types.index(memory_type)]
                        resolved_metadata = metadata[memory_type]
        
        elif resolution_type == "ltm_preferred":
            # Prefer LTM value if available
            if "ltm" in memory_types:
                preferred_memory = "ltm"
                resolved_value = values[memory_types.index("ltm")]
                resolved_metadata = metadata.get("ltm", {})
        
        else:
            # Unknown resolution type
            return False
        
        # If no resolution was determined, fail
        if preferred_memory is None or resolved_value is None:
            return False
        
        # Update all memory types with resolved value
        for memory_type in memory_types:
            if memory_type != preferred_memory and memory_type in self.memories:
                await self.memories[memory_type].update(key, resolved_value, resolved_metadata)
        
        return True
    
    async def forget(
        self,
        key_pattern: str,
        memory_types: Optional[List[str]] = None,
        older_than: Optional[float] = None
    ) -> Dict[str, int]:
        """
        Forget (delete) items matching criteria.
        
        Args:
            key_pattern: Pattern to match keys
            memory_types: Memory types to forget from (default: all)
            older_than: Timestamp threshold for forgetting
            
        Returns:
            Dictionary of memory types and count of items forgotten
        """
        memory_types = memory_types or list(self.memories.keys())
        results = {}
        
        for memory_type in memory_types:
            if memory_type not in self.memories:
                continue
            
            # Search for matching items
            matching_items = await self.memories[memory_type].search(key_pattern, limit=1000)
            
            # Filter by age if specified
            if older_than is not None:
                matching_items = [
                    item for item in matching_items
                    if "metadata" in item and "timestamp" in item["metadata"] and item["metadata"]["timestamp"] < older_than
                ]
            
            # Delete matching items
            forgotten_count = 0
            for item in matching_items:
                key = item["key"]
                success = await self.memories[memory_type].delete(key)
                if success:
                    forgotten_count += 1
            
            results[memory_type] = forgotten_count
        
        return results
    
    async def maintain(self) -> Dict[str, Any]:
        """
        Perform maintenance operations on the memory system.
        
        This includes consolidation, conflict resolution, optimization, and memory integration.
        This enhanced version implements more sophisticated memory management techniques.
        
        Returns:
            Dictionary with maintenance statistics
        """
        maintenance_stats = {
            "consolidation": {},
            "conflicts_resolved": 0,
            "optimizations": {},
            "cross_memory_integrations": 0,
            "knowledge_clusters": 0
        }
        
        # Step 1: Consolidate STM to LTM - Transfer important information to long-term storage
        consolidation_stats = await self.consolidate_stm_to_ltm()
        maintenance_stats["consolidation"]["stm_to_ltm"] = consolidation_stats
        
        # Step 2: Clear out old, low-priority items from STM to prevent overflow
        stm_prune_stats = await self._prune_stm()
        maintenance_stats["optimizations"]["stm_pruned"] = stm_prune_stats
        
        # Step 3: Detect and resolve conflicts across memory types
        # Get comprehensive sample from all memory types
        sample_keys = []
        
        # Get sample keys from STM
        stm_items = await self.stm.search("", limit=20)
        sample_keys.extend([item["key"] for item in stm_items])
        
        # Get sample keys from LTM
        ltm_items = await self.ltm.search("", limit=20)
        sample_keys.extend([item["key"] for item in ltm_items])
        
        # Get keys from working memory
        working_keys = await self.working_memory.get_all_keys()
        sample_keys.extend(working_keys)
        
        # Get recent keys from episodic memory
        episodic_items = await self.episodic.get_recent_items(10)
        sample_keys.extend([item["key"] for item in episodic_items if "key" in item])
        
        # Remove duplicates
        sample_keys = list(set(sample_keys))
        
        # Detect and resolve conflicts using multiple strategies
        conflicts_resolved = 0
        for key in sample_keys:
            conflicts = await self.detect_conflicts(key)
            for conflict in conflicts:
                # Try multiple resolution strategies in order
                resolution_methods = ["newest", "highest_priority", "ltm_preferred"]
                for method in resolution_methods:
                    success = await self.resolve_conflict(conflict, method)
                    if success:
                        conflicts_resolved += 1
                        break
        
        maintenance_stats["conflicts_resolved"] = conflicts_resolved
        
        # Step 4: Perform knowledge integration across memory types
        integration_count = await self._integrate_related_knowledge()
        maintenance_stats["cross_memory_integrations"] = integration_count
        
        # Step 5: Perform LTM clustering and consolidation
        if hasattr(self.ltm, "consolidate"):
            ltm_optimization = await self.ltm.consolidate()
            maintenance_stats["optimizations"]["ltm"] = ltm_optimization
            
        # Step 6: Identify and strengthen knowledge clusters
        clusters = await self._identify_knowledge_clusters()
        maintenance_stats["knowledge_clusters"] = len(clusters)
        
        # Step 7: Update memory indices for optimized retrieval
        await self._update_memory_indices()
        
        return maintenance_stats
        
    async def _prune_stm(self, max_age_seconds: int = 3600, keep_min_priority: int = 5) -> Dict[str, Any]:
        """
        Prune old and low-priority items from short-term memory.
        
        Args:
            max_age_seconds: Maximum age in seconds for items to keep
            keep_min_priority: Minimum priority level to keep regardless of age
            
        Returns:
            Dictionary with pruning statistics
        """
        import time
        stats = {"pruned": 0, "retained": 0}
        current_time = time.time()
        
        # Get all items from STM
        all_items = await self.stm.search("", limit=1000)
        
        for item in all_items:
            key = item["key"]
            metadata = item.get("metadata", {})
            timestamp = metadata.get("timestamp", 0)
            priority = metadata.get("priority", 0)
            
            # Skip high-priority items
            if priority >= keep_min_priority:
                stats["retained"] += 1
                continue
                
            # Prune old, low-priority items
            age = current_time - timestamp
            if age > max_age_seconds:
                await self.stm.delete(key)
                stats["pruned"] += 1
            else:
                stats["retained"] += 1
                
        return stats
        
    async def _integrate_related_knowledge(self) -> int:
        """
        Integrate related knowledge across memory types to form connections.
        
        This process identifies semantically related information across different
        memory stores and creates cross-references between them.
        
        Returns:
            Number of integrations performed
        """
        integration_count = 0
        
        # Get focus items from working memory
        focus_items = await self.working_memory.get_focus() or []
        
        for focus_item in focus_items:
            if isinstance(focus_item, dict) and "key" in focus_item:
                focus_key = focus_item["key"]
                focus_value = focus_item["value"]
                
                # Convert to string for semantic search
                if not isinstance(focus_value, str):
                    focus_value = self.ltm._value_to_string(focus_value)
                
                # Find related items in LTM based on semantic similarity
                related_items = await self.ltm.search(focus_value, limit=5)
                
                # Create cross-references
                for related_item in related_items:
                    if related_item["key"] != focus_key:
                        # Add relation metadata to both items
                        await self._create_cross_reference(focus_key, related_item["key"])
                        integration_count += 1
        
        return integration_count
    
    async def _create_cross_reference(self, key1: str, key2: str) -> bool:
        """
        Create a cross-reference between two memory items.
        
        Args:
            key1: Key of first item
            key2: Key of second item
            
        Returns:
            Success status
        """
        # Get both items
        item1_result = await self.get(key1)
        item2_result = await self.get(key2)
        
        # Skip if either item doesn't exist in any memory
        if not item1_result or not item2_result:
            return False
            
        # For each memory type where the items exist
        for memory_type, item1 in item1_result.items():
            if memory_type in item2_result:
                # Both items exist in this memory type
                item2 = item2_result[memory_type]
                
                # Add cross-reference metadata to both items
                if hasattr(self.memories[memory_type], "update"):
                    # Get existing metadata
                    if hasattr(item1, "metadata"):
                        metadata1 = item1.metadata.copy() if hasattr(item1, "metadata") else {}
                    else:
                        metadata1 = {}
                        
                    if hasattr(item2, "metadata"):
                        metadata2 = item2.metadata.copy() if hasattr(item2, "metadata") else {}
                    else:
                        metadata2 = {}
                    
                    # Add related_items list if it doesn't exist
                    if "related_items" not in metadata1:
                        metadata1["related_items"] = []
                    if "related_items" not in metadata2:
                        metadata2["related_items"] = []
                        
                    # Add cross-references
                    if key2 not in metadata1["related_items"]:
                        metadata1["related_items"].append(key2)
                    if key1 not in metadata2["related_items"]:
                        metadata2["related_items"].append(key1)
                        
                    # Update both items
                    await self.memories[memory_type].update(key1, item1, metadata1)
                    await self.memories[memory_type].update(key2, item2, metadata2)
                    
                    return True
                    
        return False

    async def _identify_knowledge_clusters(self) -> List[Dict[str, Any]]:
        """
        Identify clusters of related knowledge within the memory system.
        
        This function finds groups of semantically related information to build
        a more structured knowledge representation.
        
        Returns:
            List of knowledge clusters, each with member items and centroid
        """
        clusters = []
        
        # Start with LTM items as they tend to be more significant
        ltm_items = await self.ltm.search("", limit=100)  # Get a good sample
        
        if not ltm_items or len(ltm_items) < 5:  # Need reasonable number to cluster
            return clusters
            
        # Simple clustering approach - for each item, find related items
        processed_keys = set()
        
        for item in ltm_items:
            key = item["key"]
            
            # Skip already processed items
            if key in processed_keys:
                continue
                
            # Convert to string for semantic search
            value = item["value"]
            if not isinstance(value, str):
                value = self.ltm._value_to_string(value)
                
            # Find related items (semantic similarity)
            related_items = await self.ltm.search(value, limit=10)
            
            # Need at least 3 items to form a cluster
            if len(related_items) >= 3:
                # Mark all items as processed
                cluster_keys = [related["key"] for related in related_items]
                processed_keys.update(cluster_keys)
                
                # Create cluster
                cluster = {
                    "id": f"cluster_{len(clusters) + 1}",
                    "centroid_key": key,
                    "members": cluster_keys,
                    "size": len(related_items),
                    "semantic_similarity": sum(item.get("similarity", 0) for item in related_items) / len(related_items)
                }
                
                clusters.append(cluster)
                
        return clusters
        
    async def _update_memory_indices(self) -> None:
        """
        Update memory indices to optimize retrieval performance.
        
        This function ensures that the search indices for different memory types
        are properly optimized and aligned.
        """
        # Update LTM indices if needed
        if hasattr(self.ltm, "_rebuild_hierarchies"):
            self.ltm._rebuild_hierarchies()
            
        # Update any in-memory indices for STM
        if hasattr(self.stm, "_reindex"):
            await self.stm._reindex()
            
        # Synchronize working memory focus with STM
        focus = await self.working_memory.get_focus()
        if focus:
            # Ensure focus items are in STM with high priority
            for item in focus:
                if isinstance(item, dict) and "key" in item and "value" in item:
                    await self.stm.add(
                        item["key"], 
                        item["value"], 
                        {"priority": 9, "from_working_memory": True}
                    )
                    
        # Update recent memory from episodic
        recent_episodes = await self.episodic.get_recent_episodes(3)  # Last 3 episodes
        for episode in recent_episodes:
            # Extract key info from episodes to keep in STM
            if "events" in episode:
                for event in episode["events"]:
                    if event.get("key") == "research_completion" and "value" in event:
                        # This is a completed research result, should be in STM
                        value = event["value"]
                        if isinstance(value, dict) and "output" in value:
                            await self.stm.add(
                                f"episodic_result_{episode.get('id', '')}",
                                value["output"],
                                {"priority": 8, "from_episodic": True, "episode_id": episode.get("id")}
                            )
    
    async def set_focus(self, keys: List[str]) -> bool:
        """
        Set focus in working memory to specific items.
        
        Args:
            keys: List of keys to focus on
            
        Returns:
            Success status
        """
        return await self.working_memory.set_focus(keys)
    
    async def record_event(
        self,
        event_type: str,
        event_data: Any,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Record an event in episodic memory.
        
        Args:
            event_type: Type of event
            event_data: Event data
            metadata: Optional metadata
            
        Returns:
            Success status
        """
        return await self.episodic.add(event_type, event_data, metadata)
    
    async def create_episode(
        self,
        label: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a new episode in episodic memory.
        
        Args:
            label: Episode label
            metadata: Optional metadata
            
        Returns:
            Episode ID
        """
        return await self.episodic.create_episode(label, metadata)
    
    async def end_episode(
        self,
        outcome: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        End the current episode in episodic memory.
        
        Args:
            outcome: Optional outcome information
            
        Returns:
            Success status
        """
        return await self.episodic.end_episode(outcome)
    
    async def get_success_patterns(self) -> List[Dict[str, Any]]:
        """
        Get patterns of successful actions from episodic memory.
        
        Returns:
            List of successful patterns with statistics
        """
        return await self.episodic.get_success_patterns()
    
    async def summarize_episodes(self, query: Optional[str] = None) -> str:
        """
        Generate a summary of episodes, optionally filtered by query.
        
        Args:
            query: Optional filter query
            
        Returns:
            Summary text
        """
        # Get episodes, filtered by query if provided
        if query:
            episode_results = await self.episodic.search(query)
            episodes = [result["episode"] for result in episode_results]
        else:
            episodes = self.episodic.episodes
        
        if not episodes:
            return "No episodes to summarize."
        
        # Count by label
        label_counts = {}
        for episode in episodes:
            label = episode["label"]
            if label in label_counts:
                label_counts[label] += 1
            else:
                label_counts[label] = 1
        
        # Calculate success rates for episodes with outcomes
        success_rates = {}
        for episode in episodes:
            if "outcome" in episode and "success" in episode["outcome"]:
                label = episode["label"]
                if label not in success_rates:
                    success_rates[label] = {"successes": 0, "total": 0}
                
                success_rates[label]["total"] += 1
                if episode["outcome"]["success"]:
                    success_rates[label]["successes"] += 1
        
        # Generate summary text
        summary = f"Episodic Memory Summary ({len(episodes)} episodes):\n\n"
        
        # Add episode type counts
        summary += "Episode Types:\n"
        for label, count in label_counts.items():
            summary += f"- {label}: {count} episodes\n"
        
        # Add success rates
        if success_rates:
            summary += "\nSuccess Rates:\n"
            for label, stats in success_rates.items():
                rate = (stats["successes"] / stats["total"]) * 100
                summary += f"- {label}: {stats['successes']}/{stats['total']} ({rate:.1f}%)\n"
        
        # Calculate time statistics
        if episodes:
            durations = []
            for episode in episodes:
                if "metadata" in episode and "duration" in episode["metadata"]:
                    durations.append(episode["metadata"]["duration"])
            
            if durations:
                avg_duration = sum(durations) / len(durations)
                summary += f"\nAverage Episode Duration: {avg_duration:.1f} seconds\n"
        
        return summary