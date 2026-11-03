import React, { useState, useEffect, useMemo, useCallback } from 'react';
import ReactDOM from 'react-dom';
import { useDesignSystemTheme, Spinner } from '@databricks/design-system';
import { deleteJson, getJson, patchJson, postJson } from '@mlflow/mlflow/src/common/utils/FetchUtils';
import type { KeyValueEntity } from '@mlflow/mlflow/src/common/types';

export type StageVerdict = 'peak' | 'good' | 'improving' | 'plateau' | 'underbaked' | 'overfit' | 'bad';

export interface VerdictOption {
  verdict: StageVerdict;
  label: string;
  shortLabel: string;
  icon: string;
  defaultRating: number;
  bg: string;
  fg: string;
  border: string;
  description: string;
}

export const VERDICT_OPTIONS: VerdictOption[] = [
  {
    verdict: 'peak',
    label: 'Peak (Best)',
    shortLabel: 'Peak',
    icon: '★',
    defaultRating: 5.0,
    bg: '#fef3c7',
    fg: '#92400e',
    border: '#f59e0b',
    description: 'Sweet-spot checkpoint: crisp 1px outlines, clean pixel lattice & faithful palette',
  },
  {
    verdict: 'good',
    label: 'Good',
    shortLabel: 'Good',
    icon: '✓',
    defaultRating: 4.0,
    bg: '#dcfce7',
    fg: '#166534',
    border: '#22c55e',
    description: 'Strong pixel-art quality and prompt adherence with only minor flaws',
  },
  {
    verdict: 'improving',
    label: 'Improving',
    shortLabel: 'Improving',
    icon: '↗',
    defaultRating: 3.5,
    bg: '#e0f2fe',
    fg: '#075985',
    border: '#38bdf8',
    description: 'Moving in the right direction compared to earlier epochs; continue training',
  },
  {
    verdict: 'plateau',
    label: 'Plateau',
    shortLabel: 'Plateau',
    icon: '→',
    defaultRating: 3.0,
    bg: '#f1f5f9',
    fg: '#334155',
    border: '#94a3b8',
    description: 'No meaningful visual improvement over recent epochs',
  },
  {
    verdict: 'underbaked',
    label: 'Underbaked',
    shortLabel: 'Underbaked',
    icon: '⏳',
    defaultRating: 2.0,
    bg: '#ffedd5',
    fg: '#9a3412',
    border: '#fb923c',
    description: 'Too early in training; blurry lattice or incomplete style adaptation',
  },
  {
    verdict: 'overfit',
    label: 'Overfit',
    shortLabel: 'Overfit',
    icon: '⚠',
    defaultRating: 2.0,
    bg: '#fee2e2',
    fg: '#991b1b',
    border: '#f87171',
    description: 'Regressing past peak: fried colours, ringing artifacts, or prompt collapse',
  },
  {
    verdict: 'bad',
    label: 'Bad / Broken',
    shortLabel: 'Bad',
    icon: '✗',
    defaultRating: 1.0,
    bg: '#fecdd3',
    fg: '#881337',
    border: '#f43f5e',
    description: 'Corrupted or unusable output',
  },
];

export const VERDICT_BY_KEY: Record<string, VerdictOption> = Object.fromEntries(
  VERDICT_OPTIONS.map((o) => [o.verdict, o]),
);

const RATING_STEPS = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0];

const NOTE_MARKER_START = '<!-- STAGE_JUDGEMENTS_START -->';
const NOTE_MARKER_END = '<!-- STAGE_JUDGEMENTS_END -->';

export const STAGE_JUDGEMENT_UPDATED_EVENT = 'spellsource:stage-judgement-updated';

export interface LiveTrialJudgementSummary {
  trial: string;
  judgement: string;
  judgementVerdict: string;
  judgementRating: string;
  judgementCheckpoint: string;
  judgementEpoch: string;
  judgementNote: string;
  bestCheckpoint: string;
  bestEpoch: string;
  bestRating: string;
  bestVerdict: string;
  bestModelUri: string;
  trajectory: string;
}

const liveJudgementCache = new Map<string, LiveTrialJudgementSummary | null>();

export const getLiveTrialJudgement = (trial: string): LiveTrialJudgementSummary | null | undefined => {
  return liveJudgementCache.get(trial);
};

const emitTrialJudgementUpdate = (trial: string, summary: LiveTrialJudgementSummary | null) => {
  liveJudgementCache.set(trial, summary);
  if (typeof window !== 'undefined') {
    window.dispatchEvent(new CustomEvent(STAGE_JUDGEMENT_UPDATED_EVENT, { detail: { trial, summary } }));
  }
};

export const useLiveTrialJudgement = (trial: string | null | undefined) => {
  const [summary, setSummary] = useState<LiveTrialJudgementSummary | null | undefined>(() =>
    trial ? liveJudgementCache.get(trial) : undefined,
  );

  useEffect(() => {
    if (!trial) return;
    setSummary(liveJudgementCache.get(trial));
    const handler = (e: Event) => {
      const ce = e as CustomEvent<{ trial: string; summary: LiveTrialJudgementSummary | null }>;
      if (ce.detail?.trial === trial) {
        setSummary(ce.detail.summary);
      }
    };
    window.addEventListener(STAGE_JUDGEMENT_UPDATED_EVENT, handler);
    return () => window.removeEventListener(STAGE_JUDGEMENT_UPDATED_EVENT, handler);
  }, [trial]);

  return summary;
};

interface ModelVersionRecord {
  version: number;
  name: string;
  runId: string;
  source: string;
  description: string;
  tags: Record<string, string>;
  checkpoint: string;
  step: number;
  epoch: string;
  snappedRank?: number;
  loraRank?: number;
}

type StageSortMode = 'chronological' | 'snapped_rank' | 'lora_rank' | 'judged_only';

const formatStageMarkdownTable = (
  trial: string,
  judgedMvs: ModelVersionRecord[],
  bestVer: number | null,
): string => {
  const lines = [
    NOTE_MARKER_START,
    `### Stage Evaluation Judgements (\`${trial}\`)`,
    '',
    '| Ver | Alias | Checkpoint | Step | Epoch | `snapped/rank` | `lora/rank` | Verdict | Rating | Notes | Judged At |',
    '|---:|:---|:---|---:|---:|---:|---:|:---|---:|:---|:---|',
  ];
  for (const mv of judgedMvs) {
    const mt = mv.tags;
    const sRank =
      mt.snapped_rank || (mv.snappedRank !== undefined ? mv.snappedRank.toFixed(3) : '-');
    const lRank = mt.lora_rank || (mv.loraRank !== undefined ? mv.loraRank.toFixed(3) : '-');
    const aliasBadge = bestVer !== null && mv.version === bestVer ? '**`@best`**' : '';
    const verdict = mt.judgement || '-';
    const rating = mt.rating ? `${mt.rating}/5` : '-';
    const notes = (mt.notes || '').replace(/\|/g, '\\|').replace(/\n/g, ' ');
    const judgedAt = (mt.judged_at || '-').slice(0, 16).replace('T', ' ');
    lines.push(
      `| \`v${mv.version}\` | ${aliasBadge} | \`${mv.checkpoint}\` | ${mv.step || '-'} | ${mv.epoch || '-'} | ${sRank} | ${lRank} | **${verdict}** | ${rating} | ${notes} | ${judgedAt} |`,
    );
  }
  lines.push(NOTE_MARKER_END);
  return lines.join('\n');
};

const mergeNoteContent = (existingNote: string | undefined, tableBlock: string): string => {
  const base = (existingNote || '').trim();
  if (base.includes(NOTE_MARKER_START) && base.includes(NOTE_MARKER_END)) {
    const before = base.split(NOTE_MARKER_START)[0].trimEnd();
    const after = base.split(NOTE_MARKER_END)[1]?.trimStart() || '';
    return [before, tableBlock, after].filter(Boolean).join('\n\n');
  }
  if (base) {
    return `${base}\n\n${tableBlock}`;
  }
  return tableBlock;
};

export interface StageJudgementModalProps {
  trial: string;
  runUuid?: string;
  initialStep?: number;
  initialCheckpoint?: string;
  availableImageSteps?: number[];
  onSelectPreviewStep?: (step: number) => void;
  onClose: () => void;
}

export const StageJudgementModal: React.FC<StageJudgementModalProps> = ({
  trial,
  runUuid,
  initialStep,
  initialCheckpoint,
  availableImageSteps = [],
  onSelectPreviewStep,
  onClose,
}) => {
  const { theme } = useDesignSystemTheme();
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [statusMessage, setStatusMessage] = useState<string | null>(null);

  const [mvs, setMvs] = useState<ModelVersionRecord[]>([]);
  const [aliases, setAliases] = useState<Record<string, number>>({});
  const [compRuns, setCompRuns] = useState<
    Array<{ runUuid: string; tags: Record<string, string>; metrics: Record<string, number> }>
  >([]);
  const [trialRunId, setTrialRunId] = useState<string | null>(null);

  const [selectedVersion, setSelectedVersion] = useState<number | null>(null);
  const [stageSort, setStageSort] = useState<StageSortMode>('chronological');

  // Form state for currently selected version
  const [verdict, setVerdict] = useState<StageVerdict>('good');
  const [rating, setRating] = useState<number>(4.0);
  const [note, setNote] = useState<string>('');
  const [markBest, setMarkBest] = useState<boolean>(false);

  const populateFormFromMv = useCallback(
    (mv: ModelVersionRecord, currentAliases: Record<string, number>) => {
      const existingVerdict = (mv.tags.judgement as StageVerdict) || 'good';
      const vOpt = VERDICT_BY_KEY[existingVerdict] || VERDICT_BY_KEY['good'];
      setVerdict(vOpt.verdict);
      setRating(mv.tags.rating ? parseFloat(mv.tags.rating) : vOpt.defaultRating);
      setNote(mv.tags.notes || '');
      setMarkBest(mv.tags.is_best === 'true' || currentAliases['best'] === mv.version);
    },
    [],
  );

  const loadData = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const escapedTrial = trial.replace(/'/g, "\\'");
      const [mvRes, rmRes, runsRes] = await Promise.all([
        getJson({
          relativeUrl: 'ajax-api/2.0/mlflow/model-versions/search',
          data: { filter: `name='${escapedTrial}'`, max_results: 500 },
        }).catch(() => ({ model_versions: [] })),
        getJson({
          relativeUrl: 'ajax-api/2.0/mlflow/registered-models/get',
          data: { name: trial },
        }).catch(() => ({ registered_model: null })),
        postJson({
          relativeUrl: 'ajax-api/2.0/mlflow/runs/search',
          data: {
            experiment_ids: ['115'],
            filter: `tags.trial = '${escapedTrial}'`,
            max_results: 20,
          },
        }).catch(() => ({ runs: [] })),
      ]);

      const aliasMap: Record<string, number> = {};
      const rawAliases = (rmRes as any)?.registered_model?.aliases || [];
      for (const al of rawAliases) {
        if (al.alias && al.version !== undefined) {
          aliasMap[al.alias] = parseInt(String(al.version), 10);
        }
      }
      setAliases(aliasMap);

      const parsedCompRuns: Array<{
        runUuid: string;
        tags: Record<string, string>;
        metrics: Record<string, number>;
      }> = [];
      for (const r of (runsRes as any)?.runs || []) {
        const rUuid = r?.info?.run_uuid || r?.info?.run_id;
        if (!rUuid) continue;
        const tMap: Record<string, string> = {};
        for (const t of r?.data?.tags || []) {
          tMap[t.key] = t.value;
        }
        const mMap: Record<string, number> = {};
        for (const m of r?.data?.metrics || []) {
          mMap[m.key] = Number(m.value);
        }
        parsedCompRuns.push({ runUuid: rUuid, tags: tMap, metrics: mMap });
      }
      setCompRuns(parsedCompRuns);

      // Fetch step-indexed metric history across comparison runs so each stage shows its snapped/rank & lora/rank
      const stepSnappedRank: Record<number, number> = {};
      const stepLoraRank: Record<number, number> = {};
      await Promise.all(
        parsedCompRuns.map(async (cr) => {
          const evalStep = parseInt(cr.tags.eval_step || cr.tags.step || '0', 10);
          if (evalStep > 0) {
            if (cr.metrics['snapped.measured-ppa-v3/rank'] !== undefined) {
              stepSnappedRank[evalStep] = cr.metrics['snapped.measured-ppa-v3/rank'];
            }
            if (cr.metrics['lora/rank'] !== undefined) {
              stepLoraRank[evalStep] = cr.metrics['lora/rank'];
            }
          }
          const [snapHist, loraHist] = await Promise.all([
            getJson({
              relativeUrl: 'ajax-api/2.0/mlflow/metrics/get-history',
              data: { run_id: cr.runUuid, metric_key: 'snapped.measured-ppa-v3/rank', max_results: 1000 },
            }).catch(() => ({ metrics: [] })),
            getJson({
              relativeUrl: 'ajax-api/2.0/mlflow/metrics/get-history',
              data: { run_id: cr.runUuid, metric_key: 'lora/rank', max_results: 1000 },
            }).catch(() => ({ metrics: [] })),
          ]);
          for (const m of (snapHist as any)?.metrics || []) {
            if (m.step && Number(m.step) > 0) {
              stepSnappedRank[Number(m.step)] = Number(m.value);
            }
          }
          for (const m of (loraHist as any)?.metrics || []) {
            if (m.step && Number(m.step) > 0) {
              stepLoraRank[Number(m.step)] = Number(m.value);
            }
          }
        }),
      );

      const rawMvs = (mvRes as any)?.model_versions || [];
      const parsedMvs: ModelVersionRecord[] = rawMvs
        .map((raw: any) => {
          const tMap: Record<string, string> = {};
          for (const t of raw.tags || []) {
            tMap[t.key] = t.value;
          }
          const ver = parseInt(String(raw.version), 10);
          const stepNum = parseInt(tMap.step || '0', 10) || 0;
          const sRank =
            tMap.snapped_rank !== undefined
              ? parseFloat(tMap.snapped_rank)
              : stepSnappedRank[stepNum];
          const lRank =
            tMap.lora_rank !== undefined ? parseFloat(tMap.lora_rank) : stepLoraRank[stepNum];
          return {
            version: ver,
            name: raw.name || trial,
            runId: raw.run_id || '',
            source: raw.source || '',
            description: raw.description || '',
            tags: tMap,
            checkpoint: tMap.checkpoint || `v${ver}`,
            step: stepNum,
            epoch: tMap.epoch || '-',
            snappedRank: sRank !== undefined && !Number.isNaN(sRank) ? sRank : undefined,
            loraRank: lRank !== undefined && !Number.isNaN(lRank) ? lRank : undefined,
          };
        })
        .sort((a: ModelVersionRecord, b: ModelVersionRecord) => a.version - b.version);

      setMvs(parsedMvs);

      const latestComp =
        parsedCompRuns.find((r) => r.tags.gallery_latest === 'true') || parsedCompRuns[0];
      const tRunId =
        latestComp?.tags.trial_run_id ||
        (parsedMvs.length > 0 ? parsedMvs[parsedMvs.length - 1].runId : null);
      setTrialRunId(tRunId && tRunId !== 'base-only' ? tRunId : null);

      if (parsedMvs.length > 0) {
        let initialMv: ModelVersionRecord | undefined;
        if (initialStep && initialStep > 0) {
          initialMv = [...parsedMvs].reverse().find((m) => m.step === initialStep);
        }
        if (!initialMv && initialCheckpoint) {
          initialMv = [...parsedMvs].reverse().find((m) => m.checkpoint === initialCheckpoint);
        }
        if (!initialMv && latestComp?.tags.checkpoint) {
          initialMv = [...parsedMvs]
            .reverse()
            .find((m) => m.checkpoint === latestComp.tags.checkpoint);
        }
        if (!initialMv) {
          initialMv = parsedMvs[parsedMvs.length - 1];
        }
        setSelectedVersion(initialMv.version);
        populateFormFromMv(initialMv, aliasMap);
      }
    } catch (e: any) {
      setError(e?.message || String(e));
    } finally {
      setLoading(false);
    }
  }, [trial, initialStep, initialCheckpoint, populateFormFromMv]);

  useEffect(() => {
    loadData();
  }, [loadData]);

  const selectedMv = useMemo(
    () => mvs.find((m) => m.version === selectedVersion) || null,
    [mvs, selectedVersion],
  );

  const handleSelectVersion = (mv: ModelVersionRecord) => {
    setSelectedVersion(mv.version);
    populateFormFromMv(mv, aliases);
    setStatusMessage(null);
  };

  const handleSelectVerdict = (vOpt: VerdictOption) => {
    setVerdict(vOpt.verdict);
    setRating(vOpt.defaultRating);
    if (vOpt.verdict === 'peak') {
      setMarkBest(true);
    }
  };

  const syncAllSummaries = async (
    updatedMvs: ModelVersionRecord[],
    targetMv: ModelVersionRecord | null,
    cleared: boolean,
  ) => {
    const judgedMvs = updatedMvs
      .filter((m) => Boolean(m.tags.judgement))
      .sort((a, b) => a.version - b.version);

    const nowIso = new Date().toISOString().replace(/\.\d{3}Z$/, 'Z');
    const nowMs = Date.now();

    if (judgedMvs.length === 0) {
      // Clear run tags when no stages remain judged
      const keysToDelete = [
        'judgement',
        'judgement_verdict',
        'judgement_rating',
        'judgement_checkpoint',
        'judgement_epoch',
        'judgement_note',
        'judged_at',
        'best_checkpoint',
        'best_epoch',
        'best_rating',
        'best_verdict',
        'best_model_uri',
        'trajectory',
      ];
      for (const comp of compRuns) {
        await Promise.all(
          keysToDelete.map((k) =>
            postJson({
              relativeUrl: 'ajax-api/2.0/mlflow/runs/delete-tag',
              data: { run_id: comp.runUuid, key: k },
            }).catch(() => undefined),
          ),
        );
      }
      emitTrialJudgementUpdate(trial, null);
      return;
    }

    const explicitBest = [...judgedMvs].reverse().find((m) => m.tags.is_best === 'true');
    const bestMv =
      explicitBest ||
      judgedMvs.reduce((best, cur) => {
        const bRating = parseFloat(best.tags.rating || '0');
        const cRating = parseFloat(cur.tags.rating || '0');
        if (cRating > bRating || (cRating === bRating && cur.version >= best.version)) {
          return cur;
        }
        return best;
      }, judgedMvs[0]);

    const bestVer = bestMv.version;
    const bestCkpt = bestMv.checkpoint;
    const bestEp = bestMv.epoch || '-';
    const bestRating = parseFloat(bestMv.tags.rating || '0');
    const bestVerdict = bestMv.tags.judgement || 'good';
    const bestUri = `models:/${trial}/${bestVer}`;

    const ljMv = judgedMvs[judgedMvs.length - 1];
    const ljVer = ljMv.version;
    const ljCkpt = ljMv.checkpoint;
    const ljEp = ljMv.epoch || '-';
    const ljRating = parseFloat(ljMv.tags.rating || '0');
    const ljVerdict = ljMv.tags.judgement || 'good';
    const ljNote = ljMv.tags.notes || '';

    let trajectory = `${ljVerdict} (${ljRating}/5 @ ep ${ljEp})`;
    if (judgedMvs.length > 1) {
      const prevMv = judgedMvs[judgedMvs.length - 2];
      const prevRating = parseFloat(prevMv.tags.rating || '0');
      if (ljRating < bestRating) {
        trajectory = `regressing (${ljVerdict} ${ljRating}/5 @ ep ${ljEp}; best ${bestRating}/5 @ ep ${bestEp})`;
      } else if (ljRating > prevRating) {
        trajectory = `improving (${prevRating}->${ljRating}/5 ${ljVerdict} @ ep ${ljEp})`;
      } else if (ljRating >= 4.0) {
        trajectory = `holding ${ljVerdict} (${ljRating}/5 @ ep ${ljEp})`;
      } else {
        trajectory = `plateau (${ljVerdict} ${ljRating}/5 @ ep ${ljEp})`;
      }
    }

    // Update aliases (@best, @latest_judged, and @<verdict>)
    const nextAliases: Record<string, number> = {
      ...aliases,
      best: bestVer,
      latest_judged: ljVer,
    };
    if (!cleared && targetMv && targetMv.tags.judgement) {
      nextAliases[targetMv.tags.judgement] = targetMv.version;
    }
    setAliases(nextAliases);

    await Promise.all([
      postJson({
        relativeUrl: 'ajax-api/2.0/mlflow/registered-models/alias',
        data: { name: trial, alias: 'best', version: String(bestVer) },
      }).catch(() => undefined),
      postJson({
        relativeUrl: 'ajax-api/2.0/mlflow/registered-models/alias',
        data: { name: trial, alias: 'latest_judged', version: String(ljVer) },
      }).catch(() => undefined),
      ...(!cleared && targetMv?.tags.judgement
        ? [
            postJson({
              relativeUrl: 'ajax-api/2.0/mlflow/registered-models/alias',
              data: {
                name: trial,
                alias: targetMv.tags.judgement,
                version: String(targetMv.version),
              },
            }).catch(() => undefined),
          ]
        : []),
    ]);

    const markdownBlock = formatStageMarkdownTable(trial, judgedMvs, bestVer);

    const targetRuns =
      compRuns.length > 0
        ? compRuns
        : runUuid
          ? [{ runUuid, tags: {}, metrics: {} }]
          : [];

    for (const comp of targetRuns) {
      const mergedNote = mergeNoteContent(comp.tags['mlflow.note.content'], markdownBlock);
      const batchTags: Array<{ key: string; value: string }> = [
        { key: 'judgement', value: `${ljVerdict} (${ljRating}/5 @ ep ${ljEp})` },
        { key: 'judgement_verdict', value: ljVerdict },
        { key: 'judgement_rating', value: String(ljRating) },
        { key: 'judgement_checkpoint', value: ljCkpt },
        { key: 'judgement_epoch', value: String(ljEp) },
        { key: 'judgement_note', value: ljNote },
        { key: 'judged_at', value: nowIso },
        { key: 'best_checkpoint', value: bestCkpt },
        { key: 'best_epoch', value: String(bestEp) },
        { key: 'best_rating', value: String(bestRating) },
        { key: 'best_verdict', value: bestVerdict },
        { key: 'best_model_uri', value: bestUri },
        { key: 'trajectory', value: trajectory },
      ];
      if (!cleared && targetMv) {
        const tNote = targetMv.tags.notes ? `: ${targetMv.tags.notes}` : '';
        batchTags.push({
          key: `judgement.${targetMv.checkpoint}`,
          value: `${targetMv.tags.judgement} (${targetMv.tags.rating}/5 @ ep ${targetMv.epoch})${tNote}`.slice(
            0,
            4900,
          ),
        });
      }

      const batchMetrics: Array<{ key: string; value: number; timestamp: number; step: number }> = [
        {
          key: 'judgement/best_rating',
          value: bestRating,
          timestamp: nowMs,
          step: bestMv.step || 0,
        },
      ];
      if (!cleared && targetMv) {
        const tRating = parseFloat(targetMv.tags.rating || String(ljRating));
        batchMetrics.push({
          key: 'judgement/rating',
          value: tRating,
          timestamp: nowMs,
          step: targetMv.step || 0,
        });
        if (targetMv.version === ljVer) {
          batchMetrics.push({
            key: 'judgement/rating',
            value: tRating,
            timestamp: nowMs,
            step: 0,
          });
        }
      }

      await Promise.all([
        postJson({
          relativeUrl: 'ajax-api/2.0/mlflow/runs/log-batch',
          data: {
            run_id: comp.runUuid,
            metrics: batchMetrics,
            tags: batchTags,
          },
        }),
        postJson({
          relativeUrl: 'ajax-api/2.0/mlflow/runs/set-tag',
          data: {
            run_id: comp.runUuid,
            key: 'mlflow.note.content',
            value: mergedNote,
          },
        }),
      ]);
    }

    if (trialRunId) {
      const trainTags: Array<{ key: string; value: string }> = [
        { key: 'best_checkpoint', value: bestCkpt },
        { key: 'best_epoch', value: String(bestEp) },
        { key: 'best_rating', value: String(bestRating) },
        { key: 'best_model_uri', value: bestUri },
        { key: 'trajectory', value: trajectory },
      ];
      if (!cleared && targetMv) {
        const tNote = targetMv.tags.notes ? `: ${targetMv.tags.notes}` : '';
        trainTags.push({
          key: `judgement.${targetMv.checkpoint}`,
          value: `${targetMv.tags.judgement} (${targetMv.tags.rating}/5 @ ep ${targetMv.epoch})${tNote}`.slice(
            0,
            4900,
          ),
        });
      }
      await postJson({
        relativeUrl: 'ajax-api/2.0/mlflow/runs/log-batch',
        data: {
          run_id: trialRunId,
          tags: trainTags,
        },
      }).catch(() => undefined);
    }

    emitTrialJudgementUpdate(trial, {
      trial,
      judgement: `${ljVerdict} (${ljRating}/5 @ ep ${ljEp})`,
      judgementVerdict: ljVerdict,
      judgementRating: String(ljRating),
      judgementCheckpoint: ljCkpt,
      judgementEpoch: String(ljEp),
      judgementNote: ljNote,
      bestCheckpoint: bestCkpt,
      bestEpoch: String(bestEp),
      bestRating: String(bestRating),
      bestVerdict,
      bestModelUri: bestUri,
      trajectory,
    });
  };

  const handleSave = async () => {
    if (!selectedMv) return;
    setSaving(true);
    setError(null);
    setStatusMessage(null);
    try {
      const nowIso = new Date().toISOString().replace(/\.\d{3}Z$/, 'Z');
      const ratingStr = String(rating);
      const nextTags: Record<string, string> = {
        ...selectedMv.tags,
        judgement: verdict,
        rating: ratingStr,
        notes: note.trim(),
        judged_at: nowIso,
        is_best: markBest ? 'true' : 'false',
      };
      if (selectedMv.snappedRank !== undefined && !nextTags.snapped_rank) {
        nextTags.snapped_rank = selectedMv.snappedRank.toFixed(3);
      }
      if (selectedMv.loraRank !== undefined && !nextTags.lora_rank) {
        nextTags.lora_rank = selectedMv.loraRank.toFixed(3);
      }

      const mvTagEntries: Array<[string, string]> = [
        ['judgement', verdict],
        ['rating', ratingStr],
        ['notes', note.trim()],
        ['judged_at', nowIso],
        ['is_best', markBest ? 'true' : 'false'],
      ];
      if (nextTags.snapped_rank) mvTagEntries.push(['snapped_rank', nextTags.snapped_rank]);
      if (nextTags.lora_rank) mvTagEntries.push(['lora_rank', nextTags.lora_rank]);

      const descNote = note.trim() ? `: ${note.trim()}` : '';
      const mvDesc = `[${verdict.toUpperCase()} ${ratingStr}/5] ep ${selectedMv.epoch} (${selectedMv.checkpoint}, snap_rank=${nextTags.snapped_rank || '-'})${descNote}`.slice(
        0,
        4900,
      );

      // Clear is_best on other versions if markBest is true
      const clearOtherBestPromises = markBest
        ? mvs
            .filter((m) => m.version !== selectedMv.version && m.tags.is_best === 'true')
            .map((m) =>
              postJson({
                relativeUrl: 'ajax-api/2.0/mlflow/model-versions/set-tag',
                data: { name: trial, version: String(m.version), key: 'is_best', value: 'false' },
              }).catch(() => undefined),
            )
        : [];

      await Promise.all([
        ...clearOtherBestPromises,
        ...mvTagEntries.map(([k, v]) =>
          postJson({
            relativeUrl: 'ajax-api/2.0/mlflow/model-versions/set-tag',
            data: { name: trial, version: String(selectedMv.version), key: k, value: v },
          }),
        ),
        patchJson({
          relativeUrl: 'ajax-api/2.0/mlflow/model-versions/update',
          data: { name: trial, version: String(selectedMv.version), description: mvDesc },
        }).catch(() => undefined),
      ]);

      const updatedTargetMv: ModelVersionRecord = {
        ...selectedMv,
        description: mvDesc,
        tags: nextTags,
      };
      const updatedMvs = mvs.map((m) => {
        if (m.version === selectedMv.version) return updatedTargetMv;
        if (markBest && m.tags.is_best === 'true') {
          return { ...m, tags: { ...m.tags, is_best: 'false' } };
        }
        return m;
      });
      setMvs(updatedMvs);

      await syncAllSummaries(updatedMvs, updatedTargetMv, false);
      setStatusMessage(
        `Saved judgement for v${selectedMv.version} (${selectedMv.checkpoint}, ep ${selectedMv.epoch}): ${verdict.toUpperCase()} (${ratingStr}/5)`,
      );
    } catch (e: any) {
      setError(e?.message || String(e));
    } finally {
      setSaving(false);
    }
  };

  const handleClearStage = async (mv: ModelVersionRecord) => {
    setSaving(true);
    setError(null);
    setStatusMessage(null);
    try {
      const keysToDelete = ['judgement', 'rating', 'notes', 'judged_at', 'is_best'];
      await Promise.all([
        ...keysToDelete.map((k) =>
          deleteJson({
            relativeUrl: 'ajax-api/2.0/mlflow/model-versions/delete-tag',
            data: { name: trial, version: String(mv.version), key: k },
          }).catch(() => undefined),
        ),
        patchJson({
          relativeUrl: 'ajax-api/2.0/mlflow/model-versions/update',
          data: { name: trial, version: String(mv.version), description: '' },
        }).catch(() => undefined),
      ]);

      const nextTags = { ...mv.tags };
      for (const k of keysToDelete) delete nextTags[k];
      const updatedMvs = mvs.map((m) =>
        m.version === mv.version ? { ...m, description: '', tags: nextTags } : m,
      );
      setMvs(updatedMvs);
      await syncAllSummaries(updatedMvs, null, true);
      setStatusMessage(`Cleared judgement on v${mv.version} (${mv.checkpoint})`);
    } catch (e: any) {
      setError(e?.message || String(e));
    } finally {
      setSaving(false);
    }
  };

  const displayedStages = useMemo(() => {
    const list = [...mvs];
    if (stageSort === 'judged_only') {
      return list.filter((m) => Boolean(m.tags.judgement)).sort((a, b) => b.version - a.version);
    }
    if (stageSort === 'snapped_rank') {
      return list.sort((a, b) => {
        const sa = a.snappedRank ?? -9999;
        const sb = b.snappedRank ?? -9999;
        return sb - sa || b.version - a.version;
      });
    }
    if (stageSort === 'lora_rank') {
      return list.sort((a, b) => {
        const la = a.loraRank ?? -9999;
        const lb = b.loraRank ?? -9999;
        return lb - la || b.version - a.version;
      });
    }
    return list.sort((a, b) => b.version - a.version);
  }, [mvs, stageSort]);

  const judgedCount = useMemo(() => mvs.filter((m) => Boolean(m.tags.judgement)).length, [mvs]);
  const bestVer = aliases['best'] || null;
  const bestMv = useMemo(
    () => (bestVer ? mvs.find((m) => m.version === bestVer) || null : null),
    [mvs, bestVer],
  );

  if (typeof document === 'undefined') return null;

  return ReactDOM.createPortal(
    <div
      onClick={onClose}
      css={{
        position: 'fixed',
        inset: 0,
        zIndex: 9999,
        backgroundColor: 'rgba(15, 23, 42, 0.55)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: 20,
      }}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        css={{
          width: 'min(980px, 95vw)',
          maxHeight: '90vh',
          display: 'flex',
          flexDirection: 'column',
          backgroundColor: theme.colors.backgroundPrimary,
          color: theme.colors.textPrimary,
          borderRadius: 10,
          border: `1px solid ${theme.colors.border}`,
          boxShadow: '0 20px 45px rgba(0,0,0,0.35)',
          overflow: 'hidden',
        }}
      >
        {/* Modal Header */}
        <div
          css={{
            padding: '14px 18px',
            borderBottom: `1px solid ${theme.colors.border}`,
            backgroundColor: theme.colors.backgroundSecondary,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            gap: 12,
          }}
        >
          <div css={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
            <div css={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
              <span css={{ fontSize: 16, fontWeight: 700 }}>⚖ Stage Evaluation Judgements — {trial}</span>
              <a
                href={`#/models/${encodeURIComponent(trial)}`}
                css={{
                  fontSize: 11,
                  fontWeight: 600,
                  padding: '2px 8px',
                  borderRadius: 4,
                  backgroundColor: theme.colors.backgroundPrimary,
                  border: `1px solid ${theme.colors.border}`,
                  textDecoration: 'none',
                  color: theme.colors.actionPrimaryBackgroundDefault,
                }}
              >
                Open Model Registry ({mvs.length} stages) ↗
              </a>
              {bestMv && (
                <span
                  css={{
                    fontSize: 11,
                    fontWeight: 700,
                    padding: '2px 8px',
                    borderRadius: 4,
                    backgroundColor: '#fef3c7',
                    color: '#92400e',
                    border: '1px solid #f59e0b',
                  }}
                >
                  ★ @best: v{bestMv.version} ({bestMv.checkpoint}, ep {bestMv.epoch}) — models:/{trial}/{bestMv.version}
                </span>
              )}
            </div>
            <div css={{ fontSize: 12, color: theme.colors.textSecondary }}>
              Record human judgements at specific training stages. Saves to ModelVersion tags, updates the{' '}
              <code>@best</code> alias, plots <code>judgement/rating</code> on step charts, and updates the comparison
              gallery.
            </div>
          </div>
          <button
            type="button"
            onClick={onClose}
            css={{
              padding: '4px 10px',
              borderRadius: 6,
              border: `1px solid ${theme.colors.border}`,
              backgroundColor: theme.colors.backgroundPrimary,
              color: theme.colors.textPrimary,
              fontSize: 13,
              fontWeight: 700,
              cursor: 'pointer',
            }}
          >
            ✕ Close
          </button>
        </div>

        {/* Modal Body */}
        <div
          css={{
            flex: 1,
            overflowY: 'auto',
            padding: 18,
            display: 'flex',
            flexDirection: 'column',
            gap: 16,
          }}
        >
          {loading ? (
            <div css={{ display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 48, gap: 10 }}>
              <Spinner />
              <span>Loading registered checkpoints and stage metrics for {trial}...</span>
            </div>
          ) : mvs.length === 0 ? (
            <div css={{ padding: 24, textAlign: 'center', color: theme.colors.textSecondary }}>
              No registered checkpoint versions found in Model Registry for <strong>{trial}</strong>.
            </div>
          ) : (
            <>
              {error && (
                <div
                  css={{
                    padding: '10px 12px',
                    borderRadius: 6,
                    backgroundColor: '#fee2e2',
                    color: '#991b1b',
                    border: '1px solid #f87171',
                    fontSize: 12,
                    fontWeight: 600,
                  }}
                >
                  Error: {error}
                </div>
              )}
              {statusMessage && (
                <div
                  css={{
                    padding: '10px 12px',
                    borderRadius: 6,
                    backgroundColor: '#dcfce7',
                    color: '#166534',
                    border: '1px solid #22c55e',
                    fontSize: 12,
                    fontWeight: 600,
                  }}
                >
                  ✓ {statusMessage}
                </div>
              )}

              {/* Step 1: Verdict & Rating Editor for Selected Stage */}
              {selectedMv && (
                <div
                  css={{
                    padding: 14,
                    borderRadius: 8,
                    border: `2px solid ${theme.colors.actionPrimaryBackgroundDefault}`,
                    backgroundColor: theme.colors.backgroundSecondary,
                    display: 'flex',
                    flexDirection: 'column',
                    gap: 12,
                  }}
                >
                  <div
                    css={{
                      display: 'flex',
                      flexWrap: 'wrap',
                      alignItems: 'center',
                      justifyContent: 'space-between',
                      gap: 8,
                    }}
                  >
                    <div css={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
                      <span css={{ fontSize: 14, fontWeight: 700 }}>
                        Judging Stage: v{selectedMv.version} — <code>{selectedMv.checkpoint}</code>
                      </span>
                      <span
                        css={{
                          padding: '2px 8px',
                          borderRadius: 4,
                          fontSize: 11,
                          fontWeight: 700,
                          backgroundColor: '#ede9fe',
                          color: '#4c1d95',
                          border: '1px solid #c4b5fd',
                        }}
                      >
                        Epoch {selectedMv.epoch} (step {selectedMv.step})
                      </span>
                      {selectedMv.snappedRank !== undefined && (
                        <span
                          css={{
                            padding: '2px 7px',
                            borderRadius: 4,
                            fontSize: 11,
                            fontWeight: 600,
                            backgroundColor: theme.colors.backgroundPrimary,
                            border: `1px solid ${theme.colors.border}`,
                          }}
                        >
                          snapped/rank: {selectedMv.snappedRank.toFixed(3)}
                        </span>
                      )}
                      {selectedMv.loraRank !== undefined && (
                        <span
                          css={{
                            padding: '2px 7px',
                            borderRadius: 4,
                            fontSize: 11,
                            fontWeight: 600,
                            backgroundColor: theme.colors.backgroundPrimary,
                            border: `1px solid ${theme.colors.border}`,
                          }}
                        >
                          lora/rank: {selectedMv.loraRank.toFixed(3)}
                        </span>
                      )}
                    </div>
                    <div css={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                      {onSelectPreviewStep && availableImageSteps.includes(selectedMv.step) && (
                        <button
                          type="button"
                          onClick={() => {
                            onSelectPreviewStep(selectedMv.step);
                            setStatusMessage(
                              `Switched gallery column preview to step ${selectedMv.step} (ep ${selectedMv.epoch})`,
                            );
                          }}
                          css={{
                            padding: '3px 9px',
                            borderRadius: 5,
                            fontSize: 11,
                            fontWeight: 600,
                            cursor: 'pointer',
                            border: `1px solid ${theme.colors.actionPrimaryBackgroundDefault}`,
                            backgroundColor: theme.colors.backgroundPrimary,
                            color: theme.colors.actionPrimaryBackgroundDefault,
                          }}
                        >
                          👁 Preview Step {selectedMv.step} in Gallery
                        </button>
                      )}
                      <button
                        type="button"
                        onClick={() => {
                          navigator.clipboard?.writeText(`models:/${trial}/${selectedMv.version}`);
                          setStatusMessage(`Copied models:/${trial}/${selectedMv.version} to clipboard`);
                        }}
                        css={{
                          padding: '3px 9px',
                          borderRadius: 5,
                          fontSize: 11,
                          fontWeight: 600,
                          cursor: 'pointer',
                          border: `1px solid ${theme.colors.border}`,
                          backgroundColor: theme.colors.backgroundPrimary,
                          color: theme.colors.textPrimary,
                        }}
                      >
                        📋 Copy models:/{trial}/{selectedMv.version}
                      </button>
                    </div>
                  </div>

                  {/* Verdict Pills */}
                  <div css={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                    <span css={{ fontSize: 11, fontWeight: 700, color: theme.colors.textSecondary }}>
                      1. Choose Stage Verdict:
                    </span>
                    <div css={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(125px, 1fr))', gap: 6 }}>
                      {VERDICT_OPTIONS.map((opt) => {
                        const active = verdict === opt.verdict;
                        return (
                          <button
                            key={opt.verdict}
                            type="button"
                            onClick={() => handleSelectVerdict(opt)}
                            title={opt.description}
                            css={{
                              padding: '7px 10px',
                              borderRadius: 6,
                              cursor: 'pointer',
                              textAlign: 'left',
                              display: 'flex',
                              flexDirection: 'column',
                              gap: 2,
                              backgroundColor: active ? opt.bg : theme.colors.backgroundPrimary,
                              color: active ? opt.fg : theme.colors.textPrimary,
                              border: active ? `2px solid ${opt.border}` : `1px solid ${theme.colors.border}`,
                              boxShadow: active ? '0 1px 4px rgba(0,0,0,0.12)' : 'none',
                            }}
                          >
                            <div css={{ fontSize: 12, fontWeight: 700 }}>
                              {opt.icon} {opt.label} ({opt.defaultRating})
                            </div>
                            <div
                              css={{
                                fontSize: 10,
                                opacity: 0.85,
                                lineHeight: 1.25,
                                overflow: 'hidden',
                                display: '-webkit-box',
                                WebkitLineClamp: 2,
                                WebkitBoxOrient: 'vertical',
                              }}
                            >
                              {opt.description}
                            </div>
                          </button>
                        );
                      })}
                    </div>
                  </div>

                  {/* Numeric Rating + @best Checkbox */}
                  <div
                    css={{
                      display: 'flex',
                      flexWrap: 'wrap',
                      alignItems: 'center',
                      justifyContent: 'space-between',
                      gap: 12,
                    }}
                  >
                    <div css={{ display: 'flex', alignItems: 'center', gap: 5, flexWrap: 'wrap' }}>
                      <span css={{ fontSize: 11, fontWeight: 700, color: theme.colors.textSecondary, marginRight: 4 }}>
                        2. Numeric Rating (1–5):
                      </span>
                      {RATING_STEPS.map((r) => (
                        <button
                          key={r}
                          type="button"
                          onClick={() => setRating(r)}
                          css={{
                            padding: '3px 8px',
                            borderRadius: 4,
                            fontSize: 11,
                            fontWeight: 700,
                            cursor: 'pointer',
                            border: `1px solid ${
                              rating === r ? theme.colors.actionPrimaryBackgroundDefault : theme.colors.border
                            }`,
                            backgroundColor:
                              rating === r
                                ? theme.colors.actionPrimaryBackgroundDefault
                                : theme.colors.backgroundPrimary,
                            color: rating === r ? '#ffffff' : theme.colors.textPrimary,
                          }}
                        >
                          {r.toFixed(1)}
                        </button>
                      ))}
                    </div>

                    <label
                      css={{
                        display: 'inline-flex',
                        alignItems: 'center',
                        gap: 6,
                        fontSize: 12,
                        fontWeight: 700,
                        cursor: 'pointer',
                        padding: '4px 10px',
                        borderRadius: 5,
                        backgroundColor: markBest ? '#fef3c7' : theme.colors.backgroundPrimary,
                        color: markBest ? '#92400e' : theme.colors.textPrimary,
                        border: `1px solid ${markBest ? '#f59e0b' : theme.colors.border}`,
                      }}
                    >
                      <input
                        type="checkbox"
                        checked={markBest}
                        onChange={(e) => setMarkBest(e.target.checked)}
                      />
                      ★ Pin v{selectedMv.version} (ep {selectedMv.epoch}) as <code>@best</code> checkpoint for {trial}
                    </label>
                  </div>

                  {/* Notes + Save Action */}
                  <div css={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                    <span css={{ fontSize: 11, fontWeight: 700, color: theme.colors.textSecondary }}>
                      3. Qualitative Notes (pixel lattice, 1px outline, palette fidelity, specific prompts):
                    </span>
                    <div css={{ display: 'flex', gap: 8, alignItems: 'stretch' }}>
                      <input
                        type="text"
                        value={note}
                        onChange={(e) => setNote(e.target.value)}
                        placeholder="e.g. Crisp 1px #21215C outline on cook & pizza; best stage before fire-oven overfits..."
                        onKeyDown={(e) => {
                          if (e.key === 'Enter' && !saving) {
                            handleSave();
                          }
                        }}
                        css={{
                          flex: 1,
                          padding: '7px 10px',
                          borderRadius: 6,
                          border: `1px solid ${theme.colors.border}`,
                          backgroundColor: theme.colors.backgroundPrimary,
                          color: theme.colors.textPrimary,
                          fontSize: 12,
                        }}
                      />
                      <button
                        type="button"
                        disabled={saving}
                        onClick={handleSave}
                        css={{
                          padding: '7px 16px',
                          borderRadius: 6,
                          border: 'none',
                          backgroundColor: theme.colors.actionPrimaryBackgroundDefault,
                          color: '#ffffff',
                          fontSize: 12,
                          fontWeight: 700,
                          cursor: saving ? 'wait' : 'pointer',
                          whiteSpace: 'nowrap',
                        }}
                      >
                        {saving ? 'Saving...' : `✓ Save Judgement (v${selectedMv.version})`}
                      </button>
                      {selectedMv.tags.judgement && (
                        <button
                          type="button"
                          disabled={saving}
                          onClick={() => handleClearStage(selectedMv)}
                          css={{
                            padding: '7px 12px',
                            borderRadius: 6,
                            border: '1px solid #f87171',
                            backgroundColor: '#fee2e2',
                            color: '#991b1b',
                            fontSize: 12,
                            fontWeight: 600,
                            cursor: saving ? 'wait' : 'pointer',
                            whiteSpace: 'nowrap',
                          }}
                        >
                          Clear Stage
                        </button>
                      )}
                    </div>
                  </div>
                </div>
              )}

              {/* Step 2: All Registered Stages & Judgement History Table */}
              <div css={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                <div
                  css={{
                    display: 'flex',
                    flexWrap: 'wrap',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    gap: 8,
                  }}
                >
                  <span css={{ fontSize: 13, fontWeight: 700 }}>
                    Select Checkpoint Stage ({mvs.length} registered versions, {judgedCount} judged):
                  </span>
                  <div css={{ display: 'flex', alignItems: 'center', gap: 5, flexWrap: 'wrap' }}>
                    <span css={{ fontSize: 11, fontWeight: 600, color: theme.colors.textSecondary }}>Sort / Filter:</span>
                    {(
                      [
                        { id: 'chronological', label: 'Chronological (Newest First)' },
                        { id: 'snapped_rank', label: 'Top Snapped Rank' },
                        { id: 'lora_rank', label: 'Top LoRA Rank' },
                        { id: 'judged_only', label: `Judged Only (${judgedCount})` },
                      ] as const
                    ).map((tab) => (
                      <button
                        key={tab.id}
                        type="button"
                        onClick={() => setStageSort(tab.id)}
                        css={{
                          padding: '3px 8px',
                          borderRadius: 4,
                          fontSize: 11,
                          fontWeight: 600,
                          cursor: 'pointer',
                          border: `1px solid ${
                            stageSort === tab.id ? theme.colors.actionPrimaryBackgroundDefault : theme.colors.border
                          }`,
                          backgroundColor:
                            stageSort === tab.id
                              ? theme.colors.actionDefaultBackgroundPress
                              : theme.colors.backgroundPrimary,
                          color: theme.colors.textPrimary,
                        }}
                      >
                        {tab.label}
                      </button>
                    ))}
                  </div>
                </div>

                <div
                  css={{
                    maxHeight: 300,
                    overflowY: 'auto',
                    border: `1px solid ${theme.colors.border}`,
                    borderRadius: 6,
                  }}
                >
                  <table css={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                    <thead>
                      <tr
                        css={{
                          position: 'sticky',
                          top: 0,
                          backgroundColor: theme.colors.backgroundSecondary,
                          borderBottom: `1px solid ${theme.colors.border}`,
                          textAlign: 'left',
                        }}
                      >
                        <th css={{ padding: '6px 10px' }}>Ver</th>
                        <th css={{ padding: '6px 10px' }}>Alias</th>
                        <th css={{ padding: '6px 10px' }}>Checkpoint</th>
                        <th css={{ padding: '6px 10px' }}>Epoch</th>
                        <th css={{ padding: '6px 10px' }}>Step</th>
                        <th css={{ padding: '6px 10px' }}>snapped/rank</th>
                        <th css={{ padding: '6px 10px' }}>lora/rank</th>
                        <th css={{ padding: '6px 10px' }}>Verdict & Rating</th>
                        <th css={{ padding: '6px 10px' }}>Notes</th>
                        <th css={{ padding: '6px 10px' }}>Action</th>
                      </tr>
                    </thead>
                    <tbody>
                      {displayedStages.map((mv) => {
                        const isSelected = mv.version === selectedVersion;
                        const isBest = bestVer === mv.version;
                        const vOpt = mv.tags.judgement ? VERDICT_BY_KEY[mv.tags.judgement] : undefined;
                        return (
                          <tr
                            key={mv.version}
                            onClick={() => handleSelectVersion(mv)}
                            css={{
                              cursor: 'pointer',
                              borderBottom: `1px solid ${theme.colors.border}`,
                              backgroundColor: isSelected
                                ? theme.colors.actionDefaultBackgroundPress
                                : isBest
                                  ? 'rgba(254, 243, 199, 0.25)'
                                  : 'transparent',
                              '&:hover': {
                                backgroundColor: theme.colors.tableBackgroundUnselectedHover,
                              },
                            }}
                          >
                            <td css={{ padding: '6px 10px', fontWeight: 700 }}>v{mv.version}</td>
                            <td css={{ padding: '6px 10px' }}>
                              {isBest && (
                                <span
                                  css={{
                                    padding: '1px 6px',
                                    borderRadius: 4,
                                    fontSize: 10,
                                    fontWeight: 700,
                                    backgroundColor: '#fef3c7',
                                    color: '#92400e',
                                    border: '1px solid #f59e0b',
                                  }}
                                >
                                  ★ @best
                                </span>
                              )}
                            </td>
                            <td css={{ padding: '6px 10px', fontFamily: 'monospace', fontSize: 11 }}>
                              {mv.checkpoint}
                            </td>
                            <td css={{ padding: '6px 10px', fontWeight: 600 }}>{mv.epoch}</td>
                            <td css={{ padding: '6px 10px', color: theme.colors.textSecondary }}>{mv.step || '-'}</td>
                            <td css={{ padding: '6px 10px', fontFamily: 'monospace' }}>
                              {mv.snappedRank !== undefined ? mv.snappedRank.toFixed(3) : mv.tags.snapped_rank || '-'}
                            </td>
                            <td css={{ padding: '6px 10px', fontFamily: 'monospace' }}>
                              {mv.loraRank !== undefined ? mv.loraRank.toFixed(3) : mv.tags.lora_rank || '-'}
                            </td>
                            <td css={{ padding: '6px 10px' }}>
                              {vOpt ? (
                                <span
                                  css={{
                                    padding: '2px 7px',
                                    borderRadius: 4,
                                    fontSize: 11,
                                    fontWeight: 700,
                                    backgroundColor: vOpt.bg,
                                    color: vOpt.fg,
                                    border: `1px solid ${vOpt.border}`,
                                  }}
                                >
                                  {vOpt.icon} {vOpt.shortLabel} ({mv.tags.rating}/5)
                                </span>
                              ) : (
                                <span css={{ color: theme.colors.textSecondary, fontSize: 11 }}>—</span>
                              )}
                            </td>
                            <td
                              css={{
                                padding: '6px 10px',
                                maxWidth: 240,
                                overflow: 'hidden',
                                textOverflow: 'ellipsis',
                                whiteSpace: 'nowrap',
                              }}
                              title={mv.tags.notes || ''}
                            >
                              {mv.tags.notes || ''}
                            </td>
                            <td css={{ padding: '6px 10px' }}>
                              <button
                                type="button"
                                onClick={(e) => {
                                  e.stopPropagation();
                                  handleSelectVersion(mv);
                                }}
                                css={{
                                  padding: '2px 7px',
                                  borderRadius: 4,
                                  fontSize: 11,
                                  fontWeight: 600,
                                  cursor: 'pointer',
                                  border: `1px solid ${theme.colors.border}`,
                                  backgroundColor: isSelected
                                    ? theme.colors.actionPrimaryBackgroundDefault
                                    : theme.colors.backgroundPrimary,
                                  color: isSelected ? '#ffffff' : theme.colors.textPrimary,
                                }}
                              >
                                {isSelected ? 'Selected' : 'Select'}
                              </button>
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </div>
            </>
          )}
        </div>
      </div>
    </div>,
    document.body,
  );
};

export const StageJudgementBadgeControl: React.FC<{
  trial: string;
  runUuid?: string;
  tags?: Record<string, KeyValueEntity>;
  currentStep?: number;
  currentCheckpoint?: string;
  availableImageSteps?: number[];
  onSelectPreviewStep?: (step: number) => void;
  compact?: boolean;
}> = ({
  trial,
  runUuid,
  tags,
  currentStep,
  currentCheckpoint,
  availableImageSteps,
  onSelectPreviewStep,
  compact = false,
}) => {
  const { theme } = useDesignSystemTheme();
  const [modalOpen, setModalOpen] = useState(false);
  const liveSummary = useLiveTrialJudgement(trial);

  const verdictRaw =
    liveSummary !== undefined
      ? liveSummary?.judgementVerdict
      : tags?.['judgement_verdict']?.value || tags?.['judgement']?.value?.split(' ')[0];
  const ratingRaw =
    liveSummary !== undefined ? liveSummary?.judgementRating : tags?.['judgement_rating']?.value;
  const epochRaw =
    liveSummary !== undefined ? liveSummary?.judgementEpoch : tags?.['judgement_epoch']?.value;
  const noteRaw =
    liveSummary !== undefined ? liveSummary?.judgementNote : tags?.['judgement_note']?.value;
  const trajectoryRaw =
    liveSummary !== undefined ? liveSummary?.trajectory : tags?.['trajectory']?.value;
  const bestEpochRaw =
    liveSummary !== undefined ? liveSummary?.bestEpoch : tags?.['best_epoch']?.value;
  const bestUriRaw =
    liveSummary !== undefined ? liveSummary?.bestModelUri : tags?.['best_model_uri']?.value;

  const vOpt = verdictRaw ? VERDICT_BY_KEY[verdictRaw.toLowerCase()] : undefined;
  const isRegressing = trajectoryRaw?.startsWith('regressing') ?? false;

  const pillColors = isRegressing
    ? { bg: '#fee2e2', fg: '#991b1b', border: '#f87171' }
    : vOpt
      ? { bg: vOpt.bg, fg: vOpt.fg, border: vOpt.border }
      : {
          bg: theme.colors.backgroundSecondary,
          fg: theme.colors.textPrimary,
          border: theme.colors.border,
        };

  return (
    <>
      <div
        css={{
          display: 'inline-flex',
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: 4,
        }}
      >
        {vOpt ? (
          <button
            type="button"
            onClick={(e) => {
              e.stopPropagation();
              e.preventDefault();
              setModalOpen(true);
            }}
            title={
              [
                `Stage Verdict: ${vOpt.label} (${ratingRaw || vOpt.defaultRating}/5 @ ep ${epochRaw || '?'})`,
                trajectoryRaw ? `Trajectory: ${trajectoryRaw}` : '',
                bestUriRaw ? `Best Stage (@best): ${bestUriRaw} (ep ${bestEpochRaw || '?'})` : '',
                noteRaw ? `Note: ${noteRaw}` : '',
                'Click to edit or judge another stage',
              ]
                .filter(Boolean)
                .join('\n')
            }
            css={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: 4,
              padding: compact ? '0px 5px' : '1px 7px',
              borderRadius: 4,
              fontSize: compact ? 10 : 11,
              fontWeight: 700,
              lineHeight: compact ? '16px' : 1.4,
              cursor: 'pointer',
              backgroundColor: pillColors.bg,
              color: pillColors.fg,
              border: `1px solid ${pillColors.border}`,
              whiteSpace: 'nowrap',
            }}
          >
            <span>
              {isRegressing ? '↘' : vOpt.icon} {vOpt.shortLabel} {ratingRaw ? `${ratingRaw}/5` : ''}
              {!compact && epochRaw ? ` @ ep ${epochRaw}` : ''}
            </span>
            {!compact && isRegressing && bestEpochRaw && (
              <span css={{ opacity: 0.9 }}>(★ best ep {bestEpochRaw})</span>
            )}
            <span css={{ opacity: 0.7, fontSize: 9 }}>✎</span>
          </button>
        ) : (
          <button
            type="button"
            onClick={(e) => {
              e.stopPropagation();
              e.preventDefault();
              setModalOpen(true);
            }}
            title={`Record a stage evaluation judgement for ${trial}`}
            css={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: 3,
              padding: compact ? '0px 5px' : '1px 7px',
              borderRadius: 4,
              fontSize: compact ? 10 : 11,
              fontWeight: 600,
              lineHeight: compact ? '16px' : 1.4,
              cursor: 'pointer',
              backgroundColor: theme.colors.backgroundPrimary,
              color: theme.colors.actionPrimaryBackgroundDefault,
              border: `1px dashed ${theme.colors.actionPrimaryBackgroundDefault}`,
              whiteSpace: 'nowrap',
              '&:hover': {
                backgroundColor: theme.colors.actionDefaultBackgroundPress,
              },
            }}
          >
            {compact ? '+ ⚖ judge' : '+ ⚖ Judge Stage'}
          </button>
        )}
      </div>
      {modalOpen && (
        <StageJudgementModal
          trial={trial}
          runUuid={runUuid}
          initialStep={currentStep}
          initialCheckpoint={currentCheckpoint}
          availableImageSteps={availableImageSteps}
          onSelectPreviewStep={onSelectPreviewStep}
          onClose={() => setModalOpen(false)}
        />
      )}
    </>
  );
};
