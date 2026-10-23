import { useCallback, useEffect, useMemo, useState } from 'react';
import type { RunsChartsRunData } from '../RunsCharts.common';
import type { RunsChartCardFullScreenProps } from './ChartCard.common';
import { type RunsChartCardReorderProps, RunsChartCardWrapper, RunsChartsChartsDragGroup } from './ChartCard.common';
import { useConfirmChartCardConfigurationFn } from '../../hooks/useRunsChartsUIConfiguration';
import type { RunsChartsCardConfig, RunsChartsImageCardConfig } from '../../runs-charts.types';
import { ImageGridPlot } from '../charts/ImageGridPlot';
import { parseSpellsourceImageKey } from '../charts/ImageGridPlot.common';
import { useDesignSystemTheme } from '@databricks/design-system';
import { useImageSliderStepMarks } from '../../hooks/useImageSliderStepMarks';
import {
  DEFAULT_IMAGE_GRID_CHART_NAME,
  LOG_IMAGE_TAG_INDICATOR,
  NUM_RUNS_TO_SUPPORT_FOR_LOG_IMAGE,
} from '@mlflow/mlflow/src/experiment-tracking/constants';
import { LineSmoothSlider } from '@mlflow/mlflow/src/experiment-tracking/components/LineSmoothSlider';
import type { RunsGroupByConfig } from '@mlflow/mlflow/src/experiment-tracking/components/experiment-page/utils/experimentPage.group-row-utils';

const IMAGE_CHART_CARD_HEIGHT = 760;

export interface RunsChartsImageChartCardProps extends RunsChartCardReorderProps, RunsChartCardFullScreenProps {
  config: RunsChartsImageCardConfig;
  chartRunData: RunsChartsRunData[];

  onDelete: () => void;
  onEdit: () => void;
  groupBy: RunsGroupByConfig | null;
}

export const RunsChartsImageChartCard = ({
  config,
  chartRunData,
  onDelete,
  onEdit,
  groupBy,
  fullScreen,
  setFullScreenChart,
  ...reorderProps
}: RunsChartsImageChartCardProps) => {
  const { theme } = useDesignSystemTheme();

  // Optimizations for smoother slider experience. Maintain a local copy of config, and update
  // the global state only after the user has finished dragging the slider.
  const [tmpConfig, setTmpConfig] = useState(config);
  const confirmChartCardConfiguration = useConfirmChartCardConfigurationFn();
  const updateStep = useCallback(
    (newStep: number) => {
      // Skip updating base chart config if step is the same as current step.
      if (config.step === newStep) {
        return;
      }
      confirmChartCardConfiguration({ ...config, step: newStep } as RunsChartsImageCardConfig);
    },
    [config, confirmChartCardConfiguration],
  );
  const tmpStepChange = useCallback((newStep: number) => {
    setTmpConfig((currentConfig) => {
      // Skip updating temporary config if step is the same as current step.
      if (currentConfig.step === newStep) {
        return currentConfig;
      }
      return { ...currentConfig, step: newStep };
    });
  }, []);

  const { chartName, chartSubtitle } = useMemo(() => {
    const keys = config.imageKeys || [];
    if (keys.length === 0) {
      return { chartName: DEFAULT_IMAGE_GRID_CHART_NAME, chartSubtitle: null };
    }
    const parsed = keys.map(parseSpellsourceImageKey);
    const uniquePrompts = Array.from(new Set(parsed.map((p) => p.promptName)));
    if (uniquePrompts.length === 1 && parsed[0].promptInfo) {
      const p0 = parsed[0];
      const info = p0.promptInfo!;
      const sizeStr = p0.canvasSize ? ` (${p0.canvasSize})` : '';
      const stagesStr = parsed.map((p) => p.stageLabel.replace(/^\d+b?\.\s*/, '')).join(' → ');
      return {
        chartName: `${info.title}${sizeStr} — ${stagesStr}`,
        chartSubtitle: `${info.summary} [${p0.promptName}]`,
      };
    }
    const uniqueArms = Array.from(new Set(parsed.map((p) => p.arm)));
    if (uniqueArms.length === 1 && parsed.some((p) => p.promptInfo)) {
      const arm0 = parsed[0];
      return {
        chartName: `All Prompts Overview — ${arm0.stageLabel.replace(/^\d+b?\.\s*/, '')} (${arm0.snappedStatus})`,
        chartSubtitle: `${keys.length} prompts across models`,
      };
    }
    if (keys.length === 1) {
      return { chartName: keys[0], chartSubtitle: null };
    }
    return { chartName: DEFAULT_IMAGE_GRID_CHART_NAME, chartSubtitle: null };
  }, [config.imageKeys]);

  const toggleFullScreenChart = () => {
    setFullScreenChart?.({
      config,
      title: chartName,
      subtitle: chartSubtitle,
    });
  };

  const slicedRuns = useMemo(() => chartRunData.filter(({ hidden }) => !hidden).reverse(), [chartRunData]);

  const setCardConfig = useCallback(
    (setter: (current: RunsChartsCardConfig) => RunsChartsImageCardConfig) => {
      confirmChartCardConfiguration(setter(config));
    },
    [config, confirmChartCardConfiguration],
  );

  const { stepMarks, maxMark, minMark } = useImageSliderStepMarks({
    data: slicedRuns,
    selectedImageKeys: config.imageKeys || [],
  });

  const stepMarkLength = Object.keys(stepMarks).length;

  useEffect(() => {
    // If there is only one step mark, set the step to the min mark
    if (stepMarkLength === 1 && tmpConfig.step !== minMark) {
      updateStep(minMark);
      tmpStepChange(minMark);
    }
  }, [minMark, stepMarkLength, tmpConfig.step, updateStep, tmpStepChange]);

  const shouldDisplayImageLimitIndicator =
    slicedRuns.filter((run) => {
      return run.tags[LOG_IMAGE_TAG_INDICATOR];
    }).length > NUM_RUNS_TO_SUPPORT_FOR_LOG_IMAGE;

  const chartBody = (
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
        // The card wrapper has a definite height (ChartCard.common defaults to 360).
        // Leaving this auto outside fullscreen left the scrolling body below with no
        // height to divide, so the media sized itself from the card's width instead.
        height: '100%',
        minHeight: 0,
        width: '100%',
        overflow: 'hidden',
        marginTop: theme.spacing.sm,
        gap: theme.spacing.md,
      }}
    >
      <div
        css={{
          flex: 1,
          minHeight: 0,
          overflow: 'auto',
        }}
      >
        <ImageGridPlot
          previewData={slicedRuns}
          groupBy={groupBy}
          cardConfig={tmpConfig}
          setCardConfig={setCardConfig}
        />
      </div>
      <div
        css={{
          justifyContent: 'center',
          alignItems: 'center',
          display: 'inline-flex',
          gap: theme.spacing.md,
        }}
      >
        <div css={{ flex: 1 }}>
          <LineSmoothSlider
            value={tmpConfig.step}
            onChange={tmpStepChange}
            max={maxMark}
            min={minMark}
            marks={stepMarks}
            disabled={Object.keys(stepMarks).length <= 1}
            onAfterChange={updateStep}
            css={{
              '&[data-orientation="horizontal"]': { width: 'auto' },
            }}
          />
        </div>
      </div>
    </div>
  );

  if (fullScreen) {
    return chartBody;
  }

  const cardBodyToRender = chartBody;

  return (
    <RunsChartCardWrapper
      onEdit={onEdit}
      onDelete={onDelete}
      title={chartName}
      subtitle={
        chartSubtitle ||
        (shouldDisplayImageLimitIndicator && `Displaying images from first ${NUM_RUNS_TO_SUPPORT_FOR_LOG_IMAGE} runs`)
      }
      uuid={config.uuid}
      dragGroupKey={RunsChartsChartsDragGroup.GENERAL_AREA}
      toggleFullScreenChart={toggleFullScreenChart}
      height={IMAGE_CHART_CARD_HEIGHT}
      {...reorderProps}
    >
      {cardBodyToRender}
    </RunsChartCardWrapper>
  );
};
