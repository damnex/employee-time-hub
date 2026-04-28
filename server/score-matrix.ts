import type { RFIDEvent } from "./buffer";
import { getCurrentSession } from "./buffer";
import { faceStateMap } from "./faceState";
import { isValidTrack, trackStateMap, type TrackDirection, type TrackState } from "./trackState";

export interface ScoredTrackCandidate {
  track_id: number;
  direction: TrackDirection;
  stable: boolean;
  first_seen: number;
  last_seen: number;
  position: {
    x: number;
    y: number;
  };
  face?: {
    name?: string;
    confidence?: number;
    stable: boolean;
  };
}

export interface ScoreMatrixSnapshot {
  tags: RFIDEvent[];
  tracks: ScoredTrackCandidate[];
  scoreMatrix: number[][];
}

interface ScoreComponents {
  time_score: number;
  position_score: number;
  direction_score: number;
  face_score: number;
  final_score: number;
}

const TIME_SCORE_WINDOW_MS = 1000;
const GATE_CENTER_X = Number(process.env.SCORE_GATE_CENTER_X ?? "320");
const MAX_POSITION_DISTANCE = Number(process.env.SCORE_MAX_POSITION_DISTANCE ?? "320");
const SCORE_THRESHOLD = Number(process.env.SCORE_THRESHOLD ?? "0.5");

let latestScoreMatrixSnapshot: ScoreMatrixSnapshot = {
  tags: [],
  tracks: [],
  scoreMatrix: [],
};

function clampScore(score: number) {
  if (!Number.isFinite(score)) {
    return 0;
  }

  return Math.max(0, Math.min(1, score));
}

function cloneTag(tag: RFIDEvent): RFIDEvent {
  return { ...tag };
}

function cloneTrack(track: ScoredTrackCandidate): ScoredTrackCandidate {
  return {
    ...track,
    position: { ...track.position },
    ...(track.face
      ? {
          face: {
            ...track.face,
          },
        }
      : {}),
  };
}

function cloneSnapshot(snapshot: ScoreMatrixSnapshot): ScoreMatrixSnapshot {
  return {
    tags: snapshot.tags.map((tag) => cloneTag(tag)),
    tracks: snapshot.tracks.map((track) => cloneTrack(track)),
    scoreMatrix: snapshot.scoreMatrix.map((row) => [...row]),
  };
}

function getLastTrackPosition(track: TrackState) {
  const latestPosition = track.positions[track.positions.length - 1];
  if (!latestPosition) {
    return null;
  }

  return {
    x: latestPosition.x,
    y: latestPosition.y,
  };
}

function getUniqueRecentTags(nowMs = Date.now()) {
  const session = getCurrentSession(nowMs);
  const latestByTag = new Map<string, RFIDEvent>();

  for (const event of session.rfid) {
    const current = latestByTag.get(event.tag);
    if (!current || event.timestamp >= current.timestamp) {
      latestByTag.set(event.tag, cloneTag(event));
    }
  }

  return Array.from(latestByTag.values()).sort((left, right) => left.timestamp - right.timestamp);
}

function getValidTrackCandidates(nowMs = Date.now()) {
  const tracks: ScoredTrackCandidate[] = [];

  for (const trackState of Array.from(trackStateMap.values())) {
    if (!isValidTrack(trackState, nowMs)) {
      continue;
    }

    const position = getLastTrackPosition(trackState);
    if (!position) {
      continue;
    }

    const faceState = faceStateMap.get(trackState.track_id);
    tracks.push({
      track_id: trackState.track_id,
      direction: trackState.direction,
      stable: trackState.stable,
      first_seen: trackState.first_seen,
      last_seen: trackState.last_seen,
      position,
      ...(faceState
        ? {
            face: {
              name: faceState.name,
              confidence: faceState.confidence,
              stable: faceState.stable,
            },
          }
        : {}),
    });
  }

  return tracks.sort((left, right) => left.track_id - right.track_id);
}

export function calculateTimeSimilarity(tag: RFIDEvent, track: ScoredTrackCandidate) {
  const timeDifferenceMs = Math.abs(tag.timestamp - track.last_seen);
  return clampScore(1 - (timeDifferenceMs / TIME_SCORE_WINDOW_MS));
}

export function calculatePositionSimilarity(track: ScoredTrackCandidate) {
  const distance = Math.abs(track.position.x - GATE_CENTER_X);
  return clampScore(1 - (distance / Math.max(MAX_POSITION_DISTANCE, 1)));
}

export function calculateDirectionScore(track: ScoredTrackCandidate) {
  return track.direction === "ENTRY" || track.direction === "EXIT" ? 1 : 0;
}

export function calculateFaceScore(track: ScoredTrackCandidate) {
  return clampScore(track.face?.confidence ?? 0);
}

export function calculateMatchScore(tag: RFIDEvent, track: ScoredTrackCandidate) {
  const components: ScoreComponents = {
    time_score: calculateTimeSimilarity(tag, track),
    position_score: calculatePositionSimilarity(track),
    direction_score: calculateDirectionScore(track),
    face_score: calculateFaceScore(track),
    final_score: 0,
  };

  components.final_score = clampScore(
    components.time_score * 0.4
    + components.position_score * 0.3
    + components.direction_score * 0.2
    + components.face_score * 0.1,
  );

  return components;
}

export function buildScoreMatrix(nowMs = Date.now()): ScoreMatrixSnapshot {
  const tags = getUniqueRecentTags(nowMs);
  const tracks = getValidTrackCandidates(nowMs);
  const scoreMatrix = tags.map((tag) => {
    return tracks.map((track) => {
      const components = calculateMatchScore(tag, track);
      const acceptedScore = components.final_score >= SCORE_THRESHOLD ? components.final_score : 0;

      console.debug("[score-matrix] pair", {
        tag: tag.tag,
        track_id: track.track_id,
        time_score: Number(components.time_score.toFixed(4)),
        position_score: Number(components.position_score.toFixed(4)),
        direction_score: Number(components.direction_score.toFixed(4)),
        face_score: Number(components.face_score.toFixed(4)),
        final_score: Number(components.final_score.toFixed(4)),
        accepted_score: Number(acceptedScore.toFixed(4)),
      });

      return acceptedScore;
    });
  });

  if (tags.length || tracks.length) {
    console.debug("[score-matrix] matrix", {
      tags: tags.map((tag) => tag.tag),
      tracks: tracks.map((track) => track.track_id),
      scoreMatrix: scoreMatrix.map((row) => row.map((score) => Number(score.toFixed(4)))),
    });
  }

  tags.forEach((tag, tagIndex) => {
    const row = scoreMatrix[tagIndex] ?? [];
    let bestTrackIndex = -1;
    let bestScore = 0;

    row.forEach((score, trackIndex) => {
      if (score > bestScore) {
        bestScore = score;
        bestTrackIndex = trackIndex;
      }
    });

    if (bestTrackIndex >= 0 && bestScore > 0) {
      console.debug("[score-matrix] best-candidate", {
        tag: tag.tag,
        track_id: tracks[bestTrackIndex]?.track_id,
        score: Number(bestScore.toFixed(4)),
      });
    }
  });

  return {
    tags,
    tracks,
    scoreMatrix,
  };
}

export function refreshScoreMatrix(nowMs = Date.now()) {
  latestScoreMatrixSnapshot = buildScoreMatrix(nowMs);
  return cloneSnapshot(latestScoreMatrixSnapshot);
}

export function getScoreMatrixSnapshot(nowMs = Date.now()) {
  if (!latestScoreMatrixSnapshot.tags.length && !latestScoreMatrixSnapshot.tracks.length && !latestScoreMatrixSnapshot.scoreMatrix.length) {
    latestScoreMatrixSnapshot = buildScoreMatrix(nowMs);
  }

  return cloneSnapshot(latestScoreMatrixSnapshot);
}
