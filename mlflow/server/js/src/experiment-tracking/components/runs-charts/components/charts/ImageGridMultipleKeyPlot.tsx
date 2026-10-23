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
  const [subImageSize, setSubImageSize] = useState<number>(164);

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

  const displayRuns = previewData.filter((run: RunsChartsRunData) => Object.keys(run.images).length !== 0);

  const cellSize = useMemo(() => {
    if (!containerWidth || displayRuns.length === 0) return MIN_CELL_SIZE;
    const availableWidth = containerWidth - LABEL_COLUMN_WIDTH - theme.spacing.md * (displayRuns.length + 1);
    return Math.max(MIN_CELL_SIZE, Math.floor(availableWidth / displayRuns.length));
  }, [containerWidth, displayRuns.length, theme.spacing.md]);

  const parsedKeys = useMemo(() => cardConfig.imageKeys.map(parseSpellsourceImageKey), [cardConfig.imageKeys]);

  const isSpellsourceGrid = useMemo(
    () =>
      parsedKeys.some(
        (k) =>
          k.promptInfo !== null ||
          k.arm === 'base' ||
          k.arm === 'lora' ||
          k.arm === 'lora_vae' ||
          k.arm === 'snapped' ||
          k.arm.startsWith('snapped.'),
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
    return Array.from(map.values());
  }, [parsedKeys]);

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

    return (
      <div css={{ display: 'flex', flexDirection: 'column', width: '100%', height: '100%', gap: 8 }}>
        <div ref={measureRef} css={{ height: 0, overflow: 'hidden' }} />
        {/* Interactive Filter & Layout Bar */}
        <div
          css={{
            display: 'flex',
            flexWrap: 'wrap',
            alignItems: 'center',
            justifyContent: 'space-between',
            gap: 10,
            padding: '6px 10px',
            backgroundColor: theme.colors.backgroundSecondary,
            borderRadius: 6,
            border: `1px solid ${theme.colors.border}`,
          }}
        >
          <div css={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: 6 }}>
            <span css={{ fontSize: 11, fontWeight: 700, color: theme.colors.textSecondary }}>Prompts:</span>
            <button type="button" onClick={() => setPromptFilter('all')} css={pillButtonStyle(promptFilter === 'all')}>
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
              1. Epoch 0 → 2. Latest → 3. Snapped (All)
            </button>
            <button
              type="button"
              onClick={() => setStageFilter('snapped_only')}
              css={pillButtonStyle(stageFilter === 'snapped_only')}
            >
              3. Latest → Snapped (postprocessed) only
            </button>
            <button
              type="button"
              onClick={() => setStageFilter('latest_only')}
              css={pillButtonStyle(stageFilter === 'latest_only')}
            >
              2. Latest checkpoint (unsnapped) only
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
          </div>
        </div>

        {/* Unified Master Grid Table (Rows = Prompts, Columns = Models, Each Cell = Epoch 0 -> Latest -> Snapped) */}
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
                    Showing {filteredPromptGroups.length} prompts across {displayRuns.length} models
                  </div>
                </th>
                {displayRuns.map((run: RunsChartsRunData) => {
                  const meta = parseRunMetadata(run.displayName, run.tags);
                  const isBaseOnlyRun = meta?.evalKind === 'base';
                  const hasTunedVae = Object.keys(run.images).some((k) => k.endsWith('/lora_vae'));
                  const numSubCols =
                    stageFilter !== 'all_stages' ? 1 : isBaseOnlyRun ? 2 : hasTunedVae ? 4 : 3;
                  const minModelColWidth = Math.max(220, numSubCols * (subImageSize + 14) + 16);

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
                          alignItems: 'center',
                          gap: 6,
                        }}
                      >
                        {stageFilter === 'all_stages' ? (
                          isBaseOnlyRun ? (
                            <span>1. Epoch 0 (untrained) → 2. Epoch 0 → Snapped (postprocessed)</span>
                          ) : (
                            <span>
                              1. Epoch 0 (untrained) → 2. Latest ({meta?.shortEpochLabel || 'trained'})
                              {hasTunedVae ? ' (+ Tuned VAE)' : ''} → 3. Latest → Snapped (postprocessed)
                            </span>
                          )
                        ) : stageFilter === 'snapped_only' ? (
                          <span>3. Latest → Snapped (postprocessed)</span>
                        ) : stageFilter === 'latest_only' ? (
                          <span>2. Latest checkpoint ({meta?.shortEpochLabel || 'trained'}, unsnapped)</span>
                        ) : (
                          <span>1. Epoch 0 (untrained, unsnapped)</span>
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

                    {/* Model Cells: 1. Epoch 0 (untrained) | 2. Latest checkpoint | 3. Latest -> Snapped */}
                    {displayRuns.map((run: RunsChartsRunData) => {
                      const meta = parseRunMetadata(run.displayName, run.tags);
                      const isBaseOnlyRun = meta?.evalKind === 'base';
                      const shortEpoch = meta?.shortEpochLabel || 'Latest';

                      // Find which stages this run has for this prompt
                      const baseKey = `${promptName}/base`;
                      const loraKey = `${promptName}/lora`;
                      const loraVaeKey = `${promptName}/lora_vae`;
                      const snappedKey =
                        Object.keys(run.images).find(
                          (k) => k === `${promptName}/snapped` || k.startsWith(`${promptName}/snapped.`),
                        ) || `${promptName}/snapped.measured-ppa-v1`;

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
                            stageTitle: `2. Latest (${shortEpoch})`,
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
                      }

                      if (stageFilter === 'all_stages' || stageFilter === 'snapped_only') {
                        if (run.images[snappedKey]) {
                          stagesToRender.push({
                            key: snappedKey,
                            stageTitle: isBaseOnlyRun
                              ? '2. Epoch 0 → Snapped'
                              : `3. Latest (${shortEpoch}) → Snapped`,
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
                                      step={cardConfig.step}
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
                      <ImagePlotWithHistory metadataByStep={metadataByStep} step={cardConfig.step} runUuid={run.uuid} />
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
