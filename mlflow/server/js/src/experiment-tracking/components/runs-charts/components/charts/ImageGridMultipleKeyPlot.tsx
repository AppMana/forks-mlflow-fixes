import { useDesignSystemTheme, TableRow, TableHeader, TableCell, Table, Tooltip } from '@databricks/design-system';
import { useMemo, useRef, useState, useEffect } from 'react';
import type { RunsChartsImageCardConfig, RunsChartsCardConfig } from '../../runs-charts.types';
import type { RunsChartsRunData } from '../RunsCharts.common';
import {
  EmptyImageGridPlot,
  ImagePlotWithHistory,
  ImageGridRunHeader,
  parseSpellsourceImageKey,
} from './ImageGridPlot.common';
import type { ImageEntity } from '@mlflow/mlflow/src/experiment-tracking/types';
import { FormattedMessage } from 'react-intl';

const LABEL_COLUMN_WIDTH = 220;
const MIN_CELL_SIZE = 150;

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
  // We can't use useDynamicPlotSize or useResizeObserver directly on the table
  // container because cellSize changes feed back into the table layout, creating
  // a resize loop. Instead we observe a zero-height measurement div and use a
  // dead zone to ignore small fluctuations.
  const measureRef = useRef<HTMLDivElement>(null);
  const [containerWidth, setContainerWidth] = useState(0);
  const lastWidth = useRef(0);

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

  if (displayRuns.length === 0) {
    return <EmptyImageGridPlot />;
  }

  const parsedKeys = cardConfig.imageKeys.map(parseSpellsourceImageKey);
  const uniquePromptNames = Array.from(new Set(parsedKeys.map((k) => k.promptName)));
  const isSinglePromptCard = uniquePromptNames.length === 1 && parsedKeys.some((k) => k.promptInfo !== null);

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
              defaultMessage="Prompt / Stage"
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
        {parsedKeys.map((parsed) => {
          const { rawKey: imageKey, promptInfo, stageLabel, snappedStatus, isSnapped, canvasSize } = parsed;
          const openFullPrompt = () => {
            if (!promptInfo) return;
            const blob = new Blob([`${promptInfo.title}\n${'='.repeat(promptInfo.title.length)}\n\nPrompt Key: ${parsed.promptName}\nCanvas Size: ${canvasSize || 'default'}\n\nFull Prompt:\n${promptInfo.fullText}\n`], {
              type: 'text/plain;charset=utf-8',
            });
            const url = URL.createObjectURL(blob);
            window.open(url, '_blank', 'noopener,noreferrer');
          };
          return (
            <TableRow key={imageKey}>
              <TableCell css={{ minWidth: LABEL_COLUMN_WIDTH }}>
                {promptInfo ? (
                  <div css={{ whiteSpace: 'normal', display: 'flex', flexDirection: 'column', gap: 4 }}>
                    <div css={{ fontWeight: 600, fontSize: theme.typography.fontSizeMd, color: theme.colors.textPrimary }}>
                      {isSinglePromptCard ? stageLabel : `${promptInfo.title}${canvasSize ? ` (${canvasSize})` : ''}`}
                    </div>
                    {!isSinglePromptCard && (
                      <div css={{ fontSize: 11, fontWeight: 600, color: theme.colors.textSecondary }}>
                        {stageLabel}
                      </div>
                    )}
                    <div>
                      <span
                        css={{
                          display: 'inline-block',
                          padding: '1px 6px',
                          borderRadius: 4,
                          fontSize: 11,
                          fontWeight: 600,
                          backgroundColor: isSnapped
                            ? theme.colors.actionDefaultBackgroundPress
                            : theme.colors.backgroundSecondary,
                          color: theme.colors.textPrimary,
                        }}
                      >
                        {snappedStatus}
                      </span>
                    </div>
                    <div
                      css={{
                        fontSize: 11,
                        lineHeight: 1.35,
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
                            openFullPrompt();
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
                    <div css={{ fontSize: 10, color: theme.colors.textSecondary, opacity: 0.75 }}>{imageKey}</div>
                  </div>
                ) : (
                  <div style={{ whiteSpace: 'normal' }}>{imageKey}</div>
                )}
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
