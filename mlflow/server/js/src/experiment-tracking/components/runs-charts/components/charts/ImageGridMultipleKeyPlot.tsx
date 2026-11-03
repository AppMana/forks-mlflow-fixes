import { useDesignSystemTheme, TableRow, TableHeader, TableCell, Table, Tooltip } from '@databricks/design-system';
import { useMemo, useRef, useState, useEffect } from 'react';
import type { RunsChartsImageCardConfig, RunsChartsCardConfig } from '../../runs-charts.types';
import type { RunsChartsRunData } from '../RunsCharts.common';
import {
  EmptyImageGridPlot,
  ImagePlotWithHistory,
  ImageGridRunHeader,
  parseSpellsourceImageKey,
  parseRunMetadata,
  type ParsedImageKey,
  type PromptCatalogEntry,
} from './ImageGridPlot.common';
import { getLiveTrialJudgement, STAGE_JUDGEMENT_UPDATED_EVENT } from './StageJudgementModal';
import type { ImageEntity } from '@mlflow/mlflow/src/experiment-tracking/types';
import { FormattedMessage } from 'react-intl';

const LABEL_COLUMN_WIDTH = 250;
const MIN_CELL_SIZE = 150;

interface PromptRowGroup {
  promptName: string;
  stem: string;
  canvasSize: string | null;
  promptInfo: PromptCatalogEntry | null;
  byArm: Record<string, ParsedImageKey>;
  orderedItems: ParsedImageKey[];
}

type PromptFilterMode = 'all' | 'primary_1024' | 'dinky_1024' | 'validation_1024' | 'native_512_256' | 'generic';
type StageFilterMode = 'all_stages' | 'snapped_only' | 'latest_only' | 'epoch0_only';
type ModelFilterMode =
  | 'default_visible'
  | 'flagships_100ep'
  | 'live_training'
  | 'photoreal_edit'
  | 'edit_all'
  | 't2i_all'
  | 'all_checked';
type JudgementFilterMode = 'all' | 'peak_good' | 'improving' | 'regressing' | 'unjudged';

const SPELLSOURCE_TRAINING_EXPERIMENTS = [
  { id: '110', label: 'Qwen 2.1 (Exp 110)', live: true },
  { id: '114', label: 'Flux 2 (Exp 114)', live: true },
  { id: '157', label: 'Fibo 2 (Exp 157)', live: true },
  { id: '111', label: 'Krea 2 (Exp 111)', live: false },
  { id: '118', label: 'Iris 3B (Exp 118)', live: false },
  { id: '113', label: 'Ideogram 4 (Exp 113)', live: false },
  { id: '122', label: 'PixelDiT (Exp 122)', live: false },
  { id: '123', label: 'MOSS-Image (Exp 123)', live: false },
  { id: '120', label: 'VAE Decoder (Exp 120)', live: false },
];

export const ImageGridMultipleKeyPlot = ({
  previewData,
  cardConfig,
}: {
  previewData: RunsChartsRunData[];
  cardConfig: RunsChartsImageCardConfig;
  groupBy?: string;
  setCardConfig?: (setter: (current: RunsChartsCardConfig) => RunsChartsImageCardConfig) => void;
}) => {
  const { theme } = useDesignSystemTheme();
  const measureRef = useRef<HTMLDivElement>(null);
  const [containerWidth, setContainerWidth] = useState(0);
  const lastWidth = useRef(0);

  const [promptFilter, setPromptFilter] = useState<PromptFilterMode>('all');
  const [stageFilter, setStageFilter] = useState<StageFilterMode>('all_stages');
  const [modelFilter, setModelFilter] = useState<ModelFilterMode>('default_visible');
  const [judgementFilter, setJudgementFilter] = useState<JudgementFilterMode>('all');
  const [subImageSize, setSubImageSize] = useState<number>(164);
  const [previewStepByRunUuid, setPreviewStepByRunUuid] = useState<Record<string, number>>({});
  const [judgementTick, setJudgementTick] = useState(0);

  useEffect(() => {
    const handler = () => setJudgementTick((t) => t + 1);
    window.addEventListener(STAGE_JUDGEMENT_UPDATED_EVENT, handler);
    return () => window.removeEventListener(STAGE_JUDGEMENT_UPDATED_EVENT, handler);
  }, []);

  useEffect(() => {
    if (!measureRef.current || !window.ResizeObserver) return;
    let rafId = 0;
    const observer = new ResizeObserver(([entry]) => {
      cancelAnimationFrame(rafId);
      rafId = requestAnimationFrame(() => {
        const newWidth = Math.floor(entry.contentRect.width);
        if (newWidth > 0 && Math.abs(newWidth - lastWidth.current) > 10) {
          lastWidth.current = newWidth;
          setContainerWidth(newWidth);
        }
      });
    });
    observer.observe(measureRef.current);
    return () => {
      cancelAnimationFrame(rafId);
      observer.disconnect();
    };
  }, []);

  const allDisplayRuns = useMemo(
    () => previewData.filter((run: RunsChartsRunData) => Object.keys(run.images).length !== 0),
    [previewData],
  );

  const availableStepsByRunUuid = useMemo(() => {
    const map: Record<string, number[]> = {};
    for (const run of allDisplayRuns) {
      const steps = new Set<number>();
      for (const [k, byFile] of Object.entries(run.images || {})) {
        if (k.endsWith('/base')) continue;
        for (const entity of Object.values(byFile || {})) {
          if (entity.step !== undefined && entity.step > 0) {
            steps.add(Number(entity.step));
          }
        }
      }
      map[run.uuid] = Array.from(steps).sort((a, b) => b - a);
    }
    return map;
  }, [allDisplayRuns]);

  const displayRuns = useMemo(() => {
    const getRunEpochs = (run: RunsChartsRunData) => {
      const evalEp = parseFloat(run.tags?.['eval_epoch']?.value || run.tags?.['epoch']?.value || '0') || 0;
      const totalEp = parseFloat(run.tags?.['train_total_epochs']?.value || '0') || 0;
      return { evalEp, totalEp, maxEp: Math.max(evalEp, totalEp) };
    };
    const filtered = allDisplayRuns.filter((run) => {
      const meta = parseRunMetadata(run.displayName, run.tags);
      if (!meta) return true;
      const { maxEp } = getRunEpochs(run);
      let passModel = true;
      if (modelFilter === 'all_checked') passModel = true;
      else if (modelFilter === 'default_visible') passModel = meta.defaultVisible;
      else if (modelFilter === 'flagships_100ep') {
        passModel = (maxEp >= 50 || meta.trainStatus === 'live') && meta.defaultVisible;
      } else if (modelFilter === 'live_training') {
        passModel = meta.trainStatus === 'live' || meta.evalKind === 'progress';
      } else if (modelFilter === 'photoreal_edit') {
        passModel = meta.conditioningLabel?.startsWith('Edit: Photoreal') ?? false;
      } else if (modelFilter === 'edit_all') {
        passModel = meta.conditioningLabel?.startsWith('Edit:') ?? false;
      } else if (modelFilter === 't2i_all') {
        passModel = meta.conditioningLabel === 'T2I' && meta.defaultVisible;
      }
      if (!passModel) return false;

      if (judgementFilter !== 'all') {
        const live = getLiveTrialJudgement(meta.modelTitle);
        const verdict = (
          live !== undefined
            ? live?.judgementVerdict
            : run.tags?.['judgement_verdict']?.value || run.tags?.['judgement']?.value?.split(' ')[0] || ''
        )?.toLowerCase();
        const trajectory = (
          live !== undefined ? live?.trajectory : run.tags?.['trajectory']?.value || ''
        )?.toLowerCase();
        if (judgementFilter === 'unjudged') {
          return !verdict;
        }
        if (judgementFilter === 'peak_good') {
          return verdict === 'peak' || verdict === 'good';
        }
        if (judgementFilter === 'improving') {
          return verdict === 'improving' || trajectory.startsWith('improving');
        }
        if (judgementFilter === 'regressing') {
          return verdict === 'overfit' || verdict === 'bad' || trajectory.startsWith('regressing');
        }
      }
      return true;
    });
    const baseList = filtered.length > 0 ? filtered : allDisplayRuns;
    // Place 100-epoch / long-training flagships (evalEpoch >= 50) and live training runs FIRST on the left
    // (sorted by evalEpoch DESC), followed by newest runs first (reversing the oldest-first slice).
    const indexed = baseList.map((run, idx) => {
      const meta = parseRunMetadata(run.displayName, run.tags);
      const { evalEp, maxEp } = getRunEpochs(run);
      const isFlagshipOrLive = maxEp >= 50 || meta?.trainStatus === 'live';
      return { run, idx, evalEp, maxEp, isFlagshipOrLive };
    });
    indexed.sort((a, b) => {
      if (a.isFlagshipOrLive !== b.isFlagshipOrLive) {
        return a.isFlagshipOrLive ? -1 : 1;
      }
      if (a.isFlagshipOrLive && b.isFlagshipOrLive && Math.abs(a.evalEp - b.evalEp) > 0.01) {
        return b.evalEp - a.evalEp;
      }
      // Within non-flagship runs, put newest first (higher original idx in reversed array)
      return b.idx - a.idx;
    });
    return indexed.map((item) => item.run);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [allDisplayRuns, modelFilter, judgementFilter, judgementTick]);

  const cellSize = useMemo(() => {
    if (!containerWidth || displayRuns.length === 0) return MIN_CELL_SIZE;
    const availableWidth = containerWidth - LABEL_COLUMN_WIDTH - theme.spacing.md * (displayRuns.length + 1);
    return Math.max(MIN_CELL_SIZE, Math.floor(availableWidth / displayRuns.length));
  }, [containerWidth, displayRuns.length, theme.spacing.md]);

  const effectiveImageKeys = useMemo(() => {
    const keySet = new Set<string>(cardConfig.imageKeys || []);
    for (const run of allDisplayRuns) {
      for (const k of Object.keys(run.images || {})) {
        const pk = parseSpellsourceImageKey(k);
        if (
          pk.promptInfo !== null ||
          pk.arm === 'base' ||
          pk.arm === 'lora' ||
          pk.arm === 'lora_vae' ||
          pk.arm === 'snapped' ||
          pk.arm.startsWith('snapped.') ||
          (pk.arm.startsWith('ref_t2i.') && !pk.arm.endsWith('.reference'))
        ) {
          keySet.add(k);
        }
      }
    }
    return Array.from(keySet);
  }, [cardConfig.imageKeys, allDisplayRuns]);

  const parsedKeys = useMemo(() => effectiveImageKeys.map(parseSpellsourceImageKey), [effectiveImageKeys]);

  const isSpellsourceGrid = useMemo(
    () =>
      parsedKeys.some(
        (k) =>
          k.promptInfo !== null ||
          k.arm === 'base' ||
          k.arm === 'lora' ||
          k.arm === 'lora_vae' ||
          k.arm === 'snapped' ||
          k.arm.startsWith('snapped.') ||
          (k.arm.startsWith('ref_t2i.') && !k.arm.endsWith('.reference')),
      ),
    [parsedKeys],
  );

  const promptGroups = useMemo<PromptRowGroup[]>(() => {
    const map = new Map<string, PromptRowGroup>();
    for (const pk of parsedKeys) {
      let grp = map.get(pk.promptName);
      if (!grp) {
        grp = {
          promptName: pk.promptName,
          stem: pk.stem,
          canvasSize: pk.canvasSize,
          promptInfo: pk.promptInfo,
          byArm: {},
          orderedItems: [],
        };
        map.set(pk.promptName, grp);
      }
      grp.byArm[pk.arm] = pk;
      grp.orderedItems.push(pk);
    }
    for (const grp of map.values()) {
      grp.orderedItems.sort((a, b) => a.stageNumber - b.stageNumber || a.arm.localeCompare(b.arm));
    }
    const allGroups = Array.from(map.values());
    if (!isSpellsourceGrid || displayRuns.length === 0) {
      return allGroups;
    }
    // Filter out prompt rows where 0 out of displayRuns have any logged images
    // (e.g. hiding 1024x1024 rows when viewing 512x512-only edit models)
    const nonEmptyGroups = allGroups.filter((grp) =>
      displayRuns.some((run) =>
        grp.orderedItems.some((item) => {
          const meta = run.images?.[item.rawKey];
          return meta && Object.keys(meta).length > 0;
        }),
      ),
    );
    return nonEmptyGroups.length > 0 ? nonEmptyGroups : allGroups;
  }, [parsedKeys, isSpellsourceGrid, displayRuns]);

  const filteredPromptGroups = useMemo(() => {
    return promptGroups.filter((g) => {
      if (promptFilter === 'all') return true;
      if (promptFilter === 'primary_1024') {
        return g.canvasSize === '1024×1024' && (g.stem.startsWith('validation_') || g.stem.startsWith('dinky_'));
      }
      if (promptFilter === 'dinky_1024') {
        return g.canvasSize === '1024×1024' && g.stem.startsWith('dinky_');
      }
      if (promptFilter === 'validation_1024') {
        return g.canvasSize === '1024×1024' && g.stem.startsWith('validation_');
      }
      if (promptFilter === 'native_512_256') {
        return g.canvasSize === '512×512' || g.canvasSize === '256×256';
      }
      if (promptFilter === 'generic') {
        return g.stem.startsWith('generic_');
      }
      return true;
    });
  }, [promptGroups, promptFilter]);

  if (displayRuns.length === 0) {
    return <EmptyImageGridPlot />;
  }

  const openFullPrompt = (grp: PromptRowGroup) => {
    if (!grp.promptInfo) return;
    const blob = new Blob(
      [
        `${grp.promptInfo.title}\n${'='.repeat(grp.promptInfo.title.length)}\n\nPrompt Key: ${grp.promptName}\nCanvas Size: ${grp.canvasSize || 'default'}\n\nFull Prompt:\n${grp.promptInfo.fullText}\n`,
      ],
      { type: 'text/plain;charset=utf-8' },
    );
    const url = URL.createObjectURL(blob);
    window.open(url, '_blank', 'noopener,noreferrer');
  };

  if (isSpellsourceGrid) {
    const pillButtonStyle = (active: boolean) => ({
      padding: '3px 9px',
      borderRadius: 5,
      fontSize: 11,
      fontWeight: 600 as const,
      cursor: 'pointer' as const,
      border: `1px solid ${active ? theme.colors.actionPrimaryBackgroundDefault : theme.colors.border}`,
      backgroundColor: active ? theme.colors.actionDefaultBackgroundPress : theme.colors.backgroundPrimary,
      color: theme.colors.textPrimary,
      whiteSpace: 'nowrap' as const,
    });

    const hasPrimary1024 = promptGroups.some(
      (g) => g.canvasSize === '1024×1024' && (g.stem.startsWith('validation_') || g.stem.startsWith('dinky_')),
    );
    const hasNative512Or256 = promptGroups.some((g) => g.canvasSize === '512×512' || g.canvasSize === '256×256');
    const hasGeneric = promptGroups.some((g) => g.stem.startsWith('generic_'));

    const defaultVisibleCount = allDisplayRuns.filter(
      (r) => parseRunMetadata(r.displayName, r.tags)?.defaultVisible !== false,
    ).length;
    const flagshipCount = allDisplayRuns.filter((r) => {
      const m = parseRunMetadata(r.displayName, r.tags);
      if (m && m.defaultVisible === false) return false;
      const evalEp = parseFloat(r.tags?.['eval_epoch']?.value || r.tags?.['epoch']?.value || '0') || 0;
      const totalEp = parseFloat(r.tags?.['train_total_epochs']?.value || '0') || 0;
      return Math.max(evalEp, totalEp) >= 50 || m?.trainStatus === 'live';
    }).length;
    const liveCount = allDisplayRuns.filter((r) => {
      const m = parseRunMetadata(r.displayName, r.tags);
      return m?.trainStatus === 'live' || m?.evalKind === 'progress';
    }).length;
    const photorealCount = allDisplayRuns.filter(
      (r) => parseRunMetadata(r.displayName, r.tags)?.conditioningLabel?.startsWith('Edit: Photoreal') ?? false,
    ).length;

    return (
      <div css={{ display: 'flex', flexDirection: 'column', width: '100%', height: '100%', gap: 8 }}>
        <div ref={measureRef} css={{ height: 0, overflow: 'hidden' }} />
        {/* Interactive Filter, Model Visibility & Training Experiments Navigation Bar */}
        <div
          css={{
            display: 'flex',
            flexDirection: 'column',
            gap: 6,
            padding: '8px 10px',
            backgroundColor: theme.colors.backgroundSecondary,
            borderRadius: 6,
            border: `1px solid ${theme.colors.border}`,
          }}
        >
          {/* Row 1: Direct links to Training Experiments + Model Visibility Presets */}
          <div
            css={{
              display: 'flex',
              flexWrap: 'wrap',
              alignItems: 'center',
              justifyContent: 'space-between',
              gap: 8,
              paddingBottom: 6,
              borderBottom: `1px dashed ${theme.colors.border}`,
            }}
          >
            <div css={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: 6 }}>
              <span css={{ fontSize: 11, fontWeight: 700, color: theme.colors.textSecondary }}>Models Shown:</span>
              <button
                type="button"
                onClick={() => setModelFilter('default_visible')}
                css={pillButtonStyle(modelFilter === 'default_visible')}
                title="Show live & current benchmark models by default (100-epoch flagships & live runs sorted first on the left)"
              >
                Live & Interesting ({defaultVisibleCount})
              </button>
              {flagshipCount > 0 && (
                <button
                  type="button"
                  onClick={() => setModelFilter('flagships_100ep')}
                  css={pillButtonStyle(modelFilter === 'flagships_100ep')}
                  title="Show 100-epoch completed & active flagship runs (e.g. krea2-s8-c512-cv2 100ep, iris-s16-c1024 100ep, qwen_image21-s8-c512-cv2-ta 100ep) plus live runs"
                >
                  ★ 100-Epoch & Live Flagships ({flagshipCount})
                </button>
              )}
              {liveCount > 0 && (
                <button
                  type="button"
                  onClick={() => setModelFilter('live_training')}
                  css={pillButtonStyle(modelFilter === 'live_training')}
                >
                  Currently Training Only ({liveCount})
                </button>
              )}
              {photorealCount > 0 && (
                <button
                  type="button"
                  onClick={() => setModelFilter('photoreal_edit')}
                  css={pillButtonStyle(modelFilter === 'photoreal_edit')}
                >
                  Photoreal Edit ({photorealCount})
                </button>
              )}
              <button
                type="button"
                onClick={() => setModelFilter('edit_all')}
                css={pillButtonStyle(modelFilter === 'edit_all')}
              >
                All Edit Models
              </button>
              <button
                type="button"
                onClick={() => setModelFilter('t2i_all')}
                css={pillButtonStyle(modelFilter === 't2i_all')}
              >
                T2I Models
              </button>
              <button
                type="button"
                onClick={() => setModelFilter('all_checked')}
                css={pillButtonStyle(modelFilter === 'all_checked')}
              >
                All Checked ({allDisplayRuns.length})
              </button>
            </div>

            <div css={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: 5 }}>
              <span css={{ fontSize: 11, fontWeight: 700, color: theme.colors.textSecondary }}>
                Training Experiments:
              </span>
              {SPELLSOURCE_TRAINING_EXPERIMENTS.map((exp) => (
                <a
                  key={exp.id}
                  href={`#/experiments/${exp.id}`}
                  css={{
                    display: 'inline-flex',
                    alignItems: 'center',
                    gap: 3,
                    padding: '2px 7px',
                    borderRadius: 4,
                    fontSize: 11,
                    fontWeight: 600,
                    textDecoration: 'none',
                    backgroundColor: exp.live ? '#dbeafe' : theme.colors.backgroundPrimary,
                    color: exp.live ? '#1e40af' : theme.colors.textPrimary,
                    border: `1px solid ${exp.live ? '#93c5fd' : theme.colors.border}`,
                    '&:hover': { textDecoration: 'underline' },
                  }}
                  title={exp.live ? `${exp.label} (has live training run)` : exp.label}
                >
                  {exp.label} ↗
                </a>
              ))}
            </div>
          </div>

          {/* Row 2: Prompts & Stages Filter */}
          <div
            css={{
              display: 'flex',
              flexWrap: 'wrap',
              alignItems: 'center',
              justifyContent: 'space-between',
              gap: 10,
            }}
          >
            <div css={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: 6 }}>
              <span css={{ fontSize: 11, fontWeight: 700, color: theme.colors.textSecondary }}>Prompts:</span>
              <button
                type="button"
                onClick={() => setPromptFilter('all')}
                css={pillButtonStyle(promptFilter === 'all')}
              >
                All ({promptGroups.length})
              </button>
              {hasPrimary1024 && (
                <>
                  <button
                    type="button"
                    onClick={() => setPromptFilter('primary_1024')}
                    css={pillButtonStyle(promptFilter === 'primary_1024')}
                  >
                    1024×1024 Primary (Validation + Dinky)
                  </button>
                  <button
                    type="button"
                    onClick={() => setPromptFilter('dinky_1024')}
                    css={pillButtonStyle(promptFilter === 'dinky_1024')}
                  >
                    Dinky Diner 1024
                  </button>
                  <button
                    type="button"
                    onClick={() => setPromptFilter('validation_1024')}
                    css={pillButtonStyle(promptFilter === 'validation_1024')}
                  >
                    Validation Sprites 1024
                  </button>
                </>
              )}
              {hasNative512Or256 && (
                <button
                  type="button"
                  onClick={() => setPromptFilter('native_512_256')}
                  css={pillButtonStyle(promptFilter === 'native_512_256')}
                >
                  Native 512×512 / 256×256
                </button>
              )}
              {hasGeneric && (
                <button
                  type="button"
                  onClick={() => setPromptFilter('generic')}
                  css={pillButtonStyle(promptFilter === 'generic')}
                >
                  Generic Adherence
                </button>
              )}
            </div>

            <div css={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: 6 }}>
              <span css={{ fontSize: 11, fontWeight: 700, color: theme.colors.textSecondary }}>Stages per Model:</span>
              <button
                type="button"
                onClick={() => setStageFilter('all_stages')}
                css={pillButtonStyle(stageFilter === 'all_stages')}
              >
                1. Epoch 0 → 2. Eval Ckpt → 3. Snapped (All)
              </button>
              <button
                type="button"
                onClick={() => setStageFilter('snapped_only')}
                css={pillButtonStyle(stageFilter === 'snapped_only')}
              >
                3. Eval Ckpt → Snapped only
              </button>
              <button
                type="button"
                onClick={() => setStageFilter('latest_only')}
                css={pillButtonStyle(stageFilter === 'latest_only')}
              >
                2. Eval Ckpt (unsnapped) only
              </button>
              <button
                type="button"
                onClick={() => setStageFilter('epoch0_only')}
                css={pillButtonStyle(stageFilter === 'epoch0_only')}
              >
                1. Epoch 0 (untrained) only
              </button>

              <span css={{ fontSize: 11, fontWeight: 700, color: theme.colors.textSecondary, marginLeft: 6 }}>
                Size:
              </span>
              {[
                { label: 'S', px: 132 },
                { label: 'M', px: 164 },
                { label: 'L', px: 220 },
              ].map((opt) => (
                <button
                  key={opt.label}
                  type="button"
                  onClick={() => setSubImageSize(opt.px)}
                  css={pillButtonStyle(subImageSize === opt.px)}
                >
                  {opt.label}
                </button>
              ))}

              <span css={{ fontSize: 11, fontWeight: 700, color: theme.colors.textSecondary, marginLeft: 6 }}>
                Judgements:
              </span>
              {(
                [
                  { id: 'all', label: 'All' },
                  { id: 'peak_good', label: '★ Peak / Good' },
                  { id: 'improving', label: '↗ Improving' },
                  { id: 'regressing', label: '⚠ Overfit / Regressing' },
                  { id: 'unjudged', label: 'Unjudged' },
                ] as const
              ).map((jf) => (
                <button
                  key={jf.id}
                  type="button"
                  onClick={() => setJudgementFilter(jf.id)}
                  css={pillButtonStyle(judgementFilter === jf.id)}
                >
                  {jf.label}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Unified Master Grid Table (Rows = Prompts, Columns = Models, Each Cell = Epoch 0 -> Eval Ckpt -> Snapped) */}
        <div
          css={{
            flex: 1,
            width: '100%',
            overflow: 'auto',
            border: `1px solid ${theme.colors.border}`,
            borderRadius: 6,
            backgroundColor: theme.colors.backgroundPrimary,
          }}
        >
          <table
            css={{
              borderCollapse: 'separate',
              borderSpacing: 0,
              width: 'max-content',
              minWidth: '100%',
            }}
          >
            <thead>
              <tr>
                <th
                  css={{
                    position: 'sticky',
                    top: 0,
                    left: 0,
                    zIndex: 5,
                    width: LABEL_COLUMN_WIDTH,
                    minWidth: LABEL_COLUMN_WIDTH,
                    maxWidth: LABEL_COLUMN_WIDTH,
                    padding: '10px 12px',
                    textAlign: 'left',
                    verticalAlign: 'bottom',
                    backgroundColor: theme.colors.backgroundPrimary,
                    borderBottom: `2px solid ${theme.colors.border}`,
                    borderRight: `2px solid ${theme.colors.border}`,
                  }}
                >
                  <div css={{ fontWeight: 700, fontSize: 13, color: theme.colors.textPrimary }}>
                    Prompt (Rows) × Model (Columns)
                  </div>
                  <div css={{ fontSize: 11, fontWeight: 400, color: theme.colors.textSecondary, marginTop: 2 }}>
                    Showing {filteredPromptGroups.length} prompts across {displayRuns.length} of {allDisplayRuns.length}{' '}
                    models
                  </div>
                </th>
                {displayRuns.map((run: RunsChartsRunData) => {
                  const meta = parseRunMetadata(run.displayName, run.tags);
                  const isBaseOnlyRun = meta?.evalKind === 'base';
                  const hasTunedVae = Object.keys(run.images).some((k) => k.endsWith('/lora_vae'));
                  const refT2iArmCount = new Set(
                    Object.keys(run.images)
                      .map((k) => k.slice(k.lastIndexOf('/') + 1))
                      .filter((a) => a.startsWith('ref_t2i.') && !a.endsWith('.reference')),
                  ).size;
                  const numSubCols =
                    stageFilter !== 'all_stages'
                      ? 1 + (stageFilter === 'latest_only' ? refT2iArmCount : 0)
                      : (isBaseOnlyRun ? 2 : hasTunedVae ? 4 : 3) + refT2iArmCount;
                  const minModelColWidth = Math.max(240, numSubCols * (subImageSize + 14) + 16);
                  const availSteps = availableStepsByRunUuid[run.uuid] || [];
                  const selectedPreviewStep = previewStepByRunUuid[run.uuid] || 0;

                  return (
                    <th
                      key={run.uuid}
                      css={{
                        position: 'sticky',
                        top: 0,
                        zIndex: 4,
                        minWidth: minModelColWidth,
                        padding: '10px 12px',
                        textAlign: 'left',
                        verticalAlign: 'top',
                        backgroundColor: theme.colors.backgroundPrimary,
                        borderBottom: `2px solid ${theme.colors.border}`,
                        borderRight: `1px solid ${theme.colors.border}`,
                      }}
                    >
                      <ImageGridRunHeader
                        displayName={run.displayName}
                        color={run.color}
                        params={run.params}
                        tags={run.tags}
                        maxParamsWidth={minModelColWidth - 16}
                        runUuid={run.uuid}
                        currentStep={
                          selectedPreviewStep > 0
                            ? selectedPreviewStep
                            : parseInt(run.tags?.['eval_step']?.value || run.tags?.['step']?.value || '0', 10) || undefined
                        }
                        availableImageSteps={availSteps}
                        onSelectPreviewStep={(step) =>
                          setPreviewStepByRunUuid((prev) => ({ ...prev, [run.uuid]: step }))
                        }
                      />
                      <div
                        css={{
                          marginTop: 6,
                          paddingTop: 5,
                          borderTop: `1px dashed ${theme.colors.border}`,
                          fontSize: 11,
                          fontWeight: 600,
                          color: theme.colors.textSecondary,
                          display: 'flex',
                          flexWrap: 'wrap',
                          alignItems: 'center',
                          justifyContent: 'space-between',
                          gap: 6,
                        }}
                      >
                        <div>
                          {stageFilter === 'all_stages' ? (
                            isBaseOnlyRun ? (
                              <span>1. Epoch 0 (untrained) → 2. Epoch 0 → Snapped (postprocessed)</span>
                            ) : (
                              <span>
                                1. Epoch 0 (untrained) → 2.{' '}
                                {selectedPreviewStep > 0 ? `Step ${selectedPreviewStep}` : meta?.shortEpochLabel || 'Eval ckpt'}
                                {hasTunedVae ? ' (+ Tuned VAE)' : ''} → 3. Snapped
                                {refT2iArmCount > 0 ? ` → 4. T2I→Edit (${refT2iArmCount})` : ''}
                              </span>
                            )
                          ) : stageFilter === 'snapped_only' ? (
                            <span>
                              3. {selectedPreviewStep > 0 ? `Step ${selectedPreviewStep}` : meta?.shortEpochLabel || 'Eval ckpt'}{' '}
                              → Snapped (postprocessed)
                            </span>
                          ) : stageFilter === 'latest_only' ? (
                            <span>
                              2. {selectedPreviewStep > 0 ? `Step ${selectedPreviewStep}` : meta?.shortEpochLabel || 'Eval ckpt'}{' '}
                              (unsnapped)
                            </span>
                          ) : (
                            <span>1. Epoch 0 (untrained, unsnapped)</span>
                          )}
                        </div>
                        {availSteps.length > 1 && (
                          <label
                            onClick={(e) => e.stopPropagation()}
                            css={{
                              display: 'inline-flex',
                              alignItems: 'center',
                              gap: 4,
                              fontSize: 10,
                              fontWeight: 700,
                              color: theme.colors.textPrimary,
                            }}
                            title="Scrub this model column across logged checkpoint steps"
                          >
                            <span>Stage Step:</span>
                            <select
                              value={selectedPreviewStep}
                              onChange={(e) => {
                                const val = parseInt(e.target.value, 10) || 0;
                                setPreviewStepByRunUuid((prev) => ({ ...prev, [run.uuid]: val }));
                              }}
                              css={{
                                fontSize: 10,
                                fontWeight: 600,
                                padding: '1px 4px',
                                borderRadius: 4,
                                border: `1px solid ${theme.colors.border}`,
                                backgroundColor: theme.colors.backgroundSecondary,
                                color: theme.colors.textPrimary,
                                cursor: 'pointer',
                              }}
                            >
                              <option value={0}>Latest ({meta?.shortEpochLabel || `step ${availSteps[0]}`})</option>
                              {availSteps.map((s) => (
                                <option key={s} value={s}>
                                  step {s}
                                </option>
                              ))}
                            </select>
                          </label>
                        )}
                      </div>
                    </th>
                  );
                })}
              </tr>
            </thead>
            <tbody>
              {filteredPromptGroups.map((grp) => {
                const { promptName, promptInfo, canvasSize } = grp;
                return (
                  <tr key={promptName}>
                    {/* Sticky Left Prompt Cell */}
                    <td
                      css={{
                        position: 'sticky',
                        left: 0,
                        zIndex: 3,
                        width: LABEL_COLUMN_WIDTH,
                        minWidth: LABEL_COLUMN_WIDTH,
                        maxWidth: LABEL_COLUMN_WIDTH,
                        padding: '10px 12px',
                        verticalAlign: 'top',
                        backgroundColor: theme.colors.backgroundPrimary,
                        borderBottom: `1px solid ${theme.colors.border}`,
                        borderRight: `2px solid ${theme.colors.border}`,
                      }}
                    >
                      {promptInfo ? (
                        <div css={{ whiteSpace: 'normal', display: 'flex', flexDirection: 'column', gap: 5 }}>
                          <div css={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: 5 }}>
                            <span
                              css={{
                                fontWeight: 700,
                                fontSize: theme.typography.fontSizeMd,
                                color: theme.colors.textPrimary,
                              }}
                            >
                              {promptInfo.title}
                            </span>
                            {canvasSize && (
                              <span
                                css={{
                                  display: 'inline-block',
                                  padding: '1px 6px',
                                  borderRadius: 4,
                                  fontSize: 10,
                                  fontWeight: 700,
                                  backgroundColor: theme.colors.backgroundSecondary,
                                  color: theme.colors.textSecondary,
                                }}
                              >
                                {canvasSize} canvas
                              </span>
                            )}
                          </div>
                          <div
                            css={{
                              fontSize: 11,
                              lineHeight: 1.38,
                              color: theme.colors.textSecondary,
                            }}
                          >
                            {promptInfo.summary}
                          </div>
                          <div>
                            <Tooltip
                              content={promptInfo.fullText}
                              componentId="mlflow.charts.image-plot.prompt-fulltext-tooltip"
                            >
                              <a
                                href="#full-prompt"
                                onClick={(e) => {
                                  e.preventDefault();
                                  openFullPrompt(grp);
                                }}
                                css={{
                                  fontSize: 11,
                                  fontWeight: 600,
                                  textDecoration: 'underline',
                                  cursor: 'pointer',
                                }}
                              >
                                View full prompt ↗
                              </a>
                            </Tooltip>
                          </div>
                          <div css={{ fontSize: 10, color: theme.colors.textSecondary, opacity: 0.7 }}>
                            {promptName}
                          </div>
                        </div>
                      ) : (
                        <div css={{ whiteSpace: 'normal', fontWeight: 600 }}>{promptName}</div>
                      )}
                    </td>

                    {/* Model Cells: 1. Epoch 0 (untrained) | 2. Eval checkpoint | 3. Snapped | 4. T2I->Edit */}
                    {displayRuns.map((run: RunsChartsRunData) => {
                      const meta = parseRunMetadata(run.displayName, run.tags);
                      const isBaseOnlyRun = meta?.evalKind === 'base';
                      const selectedPreviewStep = previewStepByRunUuid[run.uuid] || 0;
                      const shortEpoch =
                        selectedPreviewStep > 0 ? `Step ${selectedPreviewStep}` : meta?.shortEpochLabel || 'Latest';

                      // Find which stages this run has for this prompt
                      const baseKey = `${promptName}/base`;
                      const loraKey = `${promptName}/lora`;
                      const loraVaeKey = `${promptName}/lora_vae`;
                      const snappedCandidates = Object.keys(run.images)
                        .filter((k) => k === `${promptName}/snapped` || k.startsWith(`${promptName}/snapped.`))
                        .sort();
                      const snappedKey =
                        snappedCandidates[snappedCandidates.length - 1] || `${promptName}/snapped.measured-ppa-v3`;
                      const refT2iKeys = Object.keys(run.images)
                        .filter((k) => k.startsWith(`${promptName}/ref_t2i.`) && !k.endsWith('.reference'))
                        .sort();

                      const stagesToRender: Array<{
                        key: string;
                        stageTitle: string;
                        badgeText: string;
                        isSnapped: boolean;
                      }> = [];

                      if (stageFilter === 'all_stages' || stageFilter === 'epoch0_only') {
                        if (run.images[baseKey]) {
                          stagesToRender.push({
                            key: baseKey,
                            stageTitle: '1. Epoch 0 (untrained)',
                            badgeText: 'Unsnapped',
                            isSnapped: false,
                          });
                        }
                      }

                      if (stageFilter === 'all_stages' || stageFilter === 'latest_only') {
                        if (run.images[loraKey]) {
                          stagesToRender.push({
                            key: loraKey,
                            stageTitle: `2. ${shortEpoch}`,
                            badgeText: 'Unsnapped',
                            isSnapped: false,
                          });
                        } else if (stageFilter === 'latest_only' && isBaseOnlyRun && run.images[baseKey]) {
                          stagesToRender.push({
                            key: baseKey,
                            stageTitle: '1. Epoch 0 (untrained base)',
                            badgeText: 'Unsnapped',
                            isSnapped: false,
                          });
                        }
                        if (run.images[loraVaeKey]) {
                          stagesToRender.push({
                            key: loraVaeKey,
                            stageTitle: `2b. Tuned VAE (${shortEpoch})`,
                            badgeText: 'Unsnapped (VAE)',
                            isSnapped: false,
                          });
                        }
                        for (const rk of refT2iKeys) {
                          const gen = rk.slice(`${promptName}/ref_t2i.`.length);
                          stagesToRender.push({
                            key: rk,
                            stageTitle: `4. T2I→Edit (${gen})`,
                            badgeText: `Unsnapped (${gen} ref)`,
                            isSnapped: false,
                          });
                        }
                      }

                      if (stageFilter === 'all_stages' || stageFilter === 'snapped_only') {
                        if (run.images[snappedKey]) {
                          stagesToRender.push({
                            key: snappedKey,
                            stageTitle: isBaseOnlyRun
                              ? '2. Epoch 0 → Snapped'
                              : `3. ${shortEpoch} → Snapped`,
                            badgeText: 'Snapped (postprocessed)',
                            isSnapped: true,
                          });
                        }
                      }

                      return (
                        <td
                          key={run.uuid}
                          css={{
                            padding: '8px 10px',
                            verticalAlign: 'top',
                            borderBottom: `1px solid ${theme.colors.border}`,
                            borderRight: `1px solid ${theme.colors.border}`,
                            '&:hover': {
                              backgroundColor: theme.colors.tableBackgroundUnselectedHover,
                            },
                          }}
                        >
                          {stagesToRender.length > 0 ? (
                            <div css={{ display: 'flex', flexDirection: 'row', alignItems: 'flex-start', gap: 10 }}>
                              {stagesToRender.map((st) => {
                                const metadataByStep = Object.values(run.images[st.key] || {}).reduce(
                                  (acc, metadata) => {
                                    if (metadata.step !== undefined) {
                                      acc[metadata.step] = metadata;
                                    }
                                    return acc;
                                  },
                                  {} as Record<number, ImageEntity>,
                                );
                                return (
                                  <div
                                    key={st.key}
                                    css={{
                                      display: 'flex',
                                      flexDirection: 'column',
                                      gap: 4,
                                      width: subImageSize,
                                      flexShrink: 0,
                                    }}
                                  >
                                    <div
                                      css={{
                                        display: 'flex',
                                        flexDirection: 'column',
                                        gap: 2,
                                        padding: '3px 6px',
                                        borderRadius: 4,
                                        backgroundColor: st.isSnapped
                                          ? theme.colors.actionDefaultBackgroundPress
                                          : theme.colors.backgroundSecondary,
                                      }}
                                    >
                                      <div
                                        css={{
                                          fontSize: 11,
                                          fontWeight: 700,
                                          color: theme.colors.textPrimary,
                                          whiteSpace: 'nowrap',
                                          overflow: 'hidden',
                                          textOverflow: 'ellipsis',
                                        }}
                                        title={st.stageTitle}
                                      >
                                        {st.stageTitle}
                                      </div>
                                      <div
                                        css={{
                                          fontSize: 10,
                                          fontWeight: 600,
                                          color: theme.colors.textSecondary,
                                          whiteSpace: 'nowrap',
                                          overflow: 'hidden',
                                          textOverflow: 'ellipsis',
                                        }}
                                      >
                                        {st.badgeText}
                                      </div>
                                    </div>
                                    <ImagePlotWithHistory
                                      metadataByStep={metadataByStep}
                                      step={
                                        st.key === baseKey
                                          ? 0
                                          : selectedPreviewStep > 0
                                            ? selectedPreviewStep
                                            : cardConfig.step
                                      }
                                      runUuid={run.uuid}
                                      imageSize={subImageSize}
                                    />
                                  </div>
                                );
                              })}
                            </div>
                          ) : (
                            <div
                              css={{
                                height: subImageSize,
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'center',
                                color: theme.colors.textSecondary,
                                fontSize: 11,
                                opacity: 0.6,
                                padding: '0 12px',
                              }}
                            >
                              Not sampled at {canvasSize || 'this size'}
                            </div>
                          )}
                        </td>
                      );
                    })}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    );
  }

  return (
    <div css={{ height: '100%', width: '100%' }}>
      <div ref={measureRef} css={{ height: 0, overflow: 'hidden' }} />
      <Table grid scrollable>
        <TableRow isHeader>
          <TableHeader
            componentId="codegen_mlflow_app_src_experiment-tracking_components_runs-charts_components_charts_imagegridmultiplekeyplot.tsx_44"
            css={{ minWidth: LABEL_COLUMN_WIDTH }}
          >
            <FormattedMessage
              defaultMessage="keys"
              description="Experiment tracking > runs charts > charts > image grid multiple key > table header text"
            />
          </TableHeader>
          {displayRuns.map((run: RunsChartsRunData) => {
            return (
              <TableHeader
                componentId="codegen_mlflow_app_src_experiment-tracking_components_runs-charts_components_charts_imagegridmultiplekeyplot.tsx_52"
                key={run.uuid}
                css={{ minWidth: cellSize, fontWeight: 'normal' }}
              >
                <ImageGridRunHeader
                  displayName={run.displayName}
                  color={run.color}
                  params={run.params}
                  tags={run.tags}
                  maxParamsWidth={cellSize}
                  showParams={cardConfig.showRunParams !== false}
                />
              </TableHeader>
            );
          })}
        </TableRow>
        {cardConfig.imageKeys.map((imageKey) => {
          return (
            <TableRow key={imageKey}>
              <TableCell css={{ minWidth: LABEL_COLUMN_WIDTH }}>
                <div style={{ whiteSpace: 'normal' }}>{imageKey}</div>
              </TableCell>
              {displayRuns.map((run: RunsChartsRunData) => {
                if (run.images[imageKey] && Object.keys(run.images[imageKey]).length > 0) {
                  const metadataByStep = Object.values(run.images[imageKey]).reduce(
                    (acc, metadata) => {
                      if (metadata.step !== undefined) {
                        acc[metadata.step] = metadata;
                      }
                      return acc;
                    },
                    {} as Record<number, ImageEntity>,
                  );
                  return (
                    <TableCell
                      key={run.uuid}
                      css={{
                        minWidth: cellSize,
                        '&:hover': {
                          backgroundColor: theme.colors.tableBackgroundUnselectedHover,
                        },
                      }}
                    >
                      <ImagePlotWithHistory
                        metadataByStep={metadataByStep}
                        step={cardConfig.step}
                        runUuid={run.uuid}
                        artifactRootUri={run.runInfo?.artifactUri ?? undefined}
                      />
                    </TableCell>
                  );
                }
                return <TableCell key={run.uuid} css={{ minWidth: cellSize }} />;
              })}
            </TableRow>
          );
        })}
      </Table>
    </div>
  );
};
