import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

import numpy as np
from dateutil import parser
from scipy.spatial.distance import cosine

logger = logging.getLogger(__name__)


@dataclass
class TemporalEvent:
    """Represents a temporal event with its properties."""
    id: str
    content: str
    start_time: datetime
    end_time: Optional[datetime]
    duration: Optional[timedelta]
    confidence: float
    metadata: Dict[str, Any]
    embeddings: List[float]


@dataclass
class TemporalSequence:
    """Represents a sequence of temporal events."""
    id: str
    events: List[TemporalEvent]
    start_time: datetime
    end_time: datetime
    duration: timedelta
    metadata: Dict[str, Any]


class TemporalReasoner:
    """Handles temporal reasoning and event sequencing."""

    def __init__(self, similarity_threshold: float = 0.8):
        self.similarity_threshold = similarity_threshold
        self.sequences: Dict[str, TemporalSequence] = {}
        self.events: Dict[str, TemporalEvent] = {}

    async def extract_temporal_events(
            self,
            content: str,
            embeddings: List[float],
            metadata: Optional[Dict[str, Any]] = None
    ) -> List[TemporalEvent]:
        """Extract temporal events from content."""
        try:
            events = []

            # Extract date/time expressions
            date_patterns = [
                r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
                r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
                r'\d{1,2}:\d{2}(?::\d{2})?(?:\s*[AP]M)?',  # Time
                r'(?:next|last|this)\s+(?:week|month|year)',
                r'(?:in|after|before)\s+\d+\s+(?:days|weeks|months|years)',
                r'(?:since|until|from|to)\s+\w+\s+\d{1,2}(?:st|nd|rd|th)?'
            ]

            for pattern in date_patterns:
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    try:
                        # Parse the date/time
                        parsed_time = parser.parse(match.group())

                        # Create event
                        event_id = f"event_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                        event = TemporalEvent(
                            id=event_id,
                            content=content,
                            start_time=parsed_time,
                            end_time=None,  # Will be updated if duration is found
                            duration=None,
                            confidence=0.8,  # Default confidence
                            metadata=metadata or {},
                            embeddings=embeddings
                        )

                        events.append(event)
                        self.events[event_id] = event
                    except Exception as e:
                        logger.warning(f"Failed to parse temporal expression: {str(e)}")

            return events
        except Exception as e:
            logger.error(f"Failed to extract temporal events: {str(e)}")
            raise

    async def create_temporal_sequence(
            self,
            events: List[TemporalEvent],
            metadata: Optional[Dict[str, Any]] = None
    ) -> TemporalSequence:
        """Create a temporal sequence from events."""
        try:
            if not events:
                raise ValueError("No events provided")

            # Sort events by start time
            sorted_events = sorted(events, key=lambda e: e.start_time)

            # Calculate sequence duration
            start_time = sorted_events[0].start_time
            end_time = sorted_events[-1].end_time or sorted_events[-1].start_time
            duration = end_time - start_time

            # Create sequence
            sequence_id = f"sequence_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            sequence = TemporalSequence(
                id=sequence_id,
                events=sorted_events,
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                metadata=metadata or {}
            )

            self.sequences[sequence_id] = sequence
            return sequence
        except Exception as e:
            logger.error(f"Failed to create temporal sequence: {str(e)}")
            raise

    async def find_related_events(
            self,
            event: TemporalEvent,
            time_window: Optional[timedelta] = None,
            max_results: int = 5
    ) -> List[TemporalEvent]:
        """Find events related to a given event within a time window."""
        try:
            related_events = []

            for other_event in self.events.values():
                if other_event.id == event.id:
                    continue

                # Check temporal proximity
                if time_window:
                    time_diff = abs((other_event.start_time - event.start_time).total_seconds())
                    if time_diff > time_window.total_seconds():
                        continue

                # Check semantic similarity
                similarity = 1 - cosine(event.embeddings, other_event.embeddings)
                if similarity >= self.similarity_threshold:
                    related_events.append((other_event, similarity))

            # Sort by similarity and return top results
            related_events.sort(key=lambda x: x[1], reverse=True)
            return [event for event, _ in related_events[:max_results]]
        except Exception as e:
            logger.error(f"Failed to find related events: {str(e)}")
            raise

    async def analyze_temporal_patterns(
            self,
            sequence_id: str
    ) -> Dict[str, Any]:
        """Analyze patterns in a temporal sequence."""
        try:
            if sequence_id not in self.sequences:
                raise ValueError("Sequence not found")

            sequence = self.sequences[sequence_id]

            # Calculate event intervals
            intervals = []
            for i in range(1, len(sequence.events)):
                interval = (sequence.events[i].start_time - sequence.events[i - 1].start_time).total_seconds()
                intervals.append(interval)

            # Calculate statistics
            if intervals:
                interval_stats = {
                    "mean": np.mean(intervals),
                    "std": np.std(intervals),
                    "min": min(intervals),
                    "max": max(intervals)
                }
            else:
                interval_stats = {}

            # Calculate event density
            total_duration = sequence.duration.total_seconds()
            event_density = len(sequence.events) / total_duration if total_duration > 0 else 0

            # Calculate semantic coherence
            similarities = []
            for i in range(1, len(sequence.events)):
                similarity = 1 - cosine(
                    sequence.events[i].embeddings,
                    sequence.events[i - 1].embeddings
                )
                similarities.append(similarity)

            semantic_coherence = np.mean(similarities) if similarities else 0.0

            return {
                "event_count": len(sequence.events),
                "duration": sequence.duration.total_seconds(),
                "event_density": event_density,
                "interval_stats": interval_stats,
                "semantic_coherence": semantic_coherence,
                "start_time": sequence.start_time.isoformat(),
                "end_time": sequence.end_time.isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to analyze temporal patterns: {str(e)}")
            raise

    async def predict_next_event(
            self,
            sequence_id: str,
            time_window: timedelta = timedelta(days=7)
    ) -> Optional[TemporalEvent]:
        """Predict the next event in a sequence."""
        try:
            if sequence_id not in self.sequences:
                raise ValueError("Sequence not found")

            sequence = self.sequences[sequence_id]
            if not sequence.events:
                return None

            # Get the last event
            last_event = sequence.events[-1]

            # Find similar events
            similar_events = await self.find_related_events(
                last_event,
                time_window=time_window
            )

            if not similar_events:
                return None

            # Calculate average time difference
            time_diffs = []
            for event in similar_events:
                time_diff = (event.start_time - last_event.start_time).total_seconds()
                time_diffs.append(time_diff)

            avg_time_diff = np.mean(time_diffs) if time_diffs else 0

            # Create predicted event
            predicted_time = last_event.start_time + timedelta(seconds=avg_time_diff)

            return TemporalEvent(
                id=f"predicted_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                content="Predicted event",
                start_time=predicted_time,
                end_time=None,
                duration=None,
                confidence=0.7,  # Lower confidence for predictions
                metadata={"is_predicted": True},
                embeddings=last_event.embeddings
            )
        except Exception as e:
            logger.error(f"Failed to predict next event: {str(e)}")
            raise


# Global temporal reasoner instance
temporal_reasoner = TemporalReasoner()
