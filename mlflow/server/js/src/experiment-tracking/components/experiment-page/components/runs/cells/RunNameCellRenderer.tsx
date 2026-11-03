import type { ICellRendererParams } from '@ag-grid-community/core';
import { Button, MinusSquareIcon, PlusSquareIcon, useDesignSystemTheme } from '@databricks/design-system';
import type { Theme } from '@emotion/react';
import React, { useMemo } from 'react';
import { Link } from '../../../../../../common/utils/RoutingUtils';
import Routes from '../../../../../routes';
import type { RunRowType } from '../../../utils/experimentPage.row-types';
import { GroupParentCellRenderer } from './GroupParentCellRenderer';
import invariant from 'invariant';
import { RunColorPill } from '../../RunColorPill';
import { useGetExperimentRunColor, useSaveExperimentRunColor } from '../../../hooks/useExperimentRunColor';
import { useExperimentViewRunsTableHeaderContext } from '../ExperimentViewRunsTableHeaderContext';
import { parseRunMetadata } from '../../../../runs-charts/components/charts/ImageGridPlot.common';
import { StageJudgementBadgeControl } from '../../../../runs-charts/components/charts/StageJudgementModal';

export interface RunNameCellRendererProps extends ICellRendererParams {
  data: RunRowType;
  isComparingRuns?: boolean;
  onExpand: (runUuid: string, childrenIds?: string[]) => void;
}

// eslint-disable-next-line react-component-name/react-component-name -- TODO(FEINF-4716)
export const RunNameCellRenderer = React.memo((props: RunNameCellRendererProps) => {
  const { theme } = useDesignSystemTheme();

  const saveRunColor = useSaveExperimentRunColor();
  const getRunColor = useGetExperimentRunColor();
  const { useGroupedValuesInCharts } = useExperimentViewRunsTableHeaderContext();

  // If we're rendering a group row, use relevant component
  if (props.data.groupParentInfo) {
    return <GroupParentCellRenderer {...props} />;
  }
  const { onExpand, data } = props;
  const { runName, experimentId, runUuid, runDateAndNestInfo, hidden, tags } = data;
  const meta = parseRunMetadata(runName || '', tags);

  // If we are not rendering a group, assert existence of necessary fields
  invariant(experimentId, 'experimentId should be set for run rows');
  invariant(runUuid, 'runUuid should be set for run rows');
  invariant(runDateAndNestInfo, 'runDateAndNestInfo should be set for run rows');

  const { hasExpander, expanderOpen, childrenIds, level, belongsToGroup } = runDateAndNestInfo;

  const renderingAsParent = !isNaN(level) && hasExpander;
  const hideRunColorControl = belongsToGroup && useGroupedValuesInCharts;

  const trainBadgeColors =
    meta?.trainStatus === 'live'
      ? { bg: '#dbeafe', fg: '#1e40af' }
      : meta?.trainStatus === 'stopped'
        ? { bg: '#fef3c7', fg: '#92400e' }
        : { bg: '#dcfce7', fg: '#166534' };

  return (
    <div css={styles.cellWrapper}>
      <div css={styles.expanderWrapper}>
        <div
          css={styles.nestLevel(theme)}
          style={{
            width: (level + 1) * theme.spacing.lg,
          }}
        >
          {renderingAsParent && (
            <Button
              componentId="codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_runs_cells_runnamecellrenderer.tsx_46"
              css={styles.expanderButton}
              size="small"
              onClick={() => {
                onExpand(runUuid, childrenIds);
              }}
              key={'Expander-' + runUuid}
              type="link"
              icon={expanderOpen ? <MinusSquareIcon /> : <PlusSquareIcon />}
            />
          )}
        </div>
      </div>
      <div css={styles.runLink}>
        {hideRunColorControl ? (
          // Render empty color pills for grouped runs
          <div css={{ width: 12, height: 12, flexShrink: 0 }} />
        ) : (
          <RunColorPill
            color={getRunColor(runUuid)}
            hidden={props.isComparingRuns && hidden}
            data-testid="experiment-view-table-run-color"
            onChangeColor={(colorValue) => saveRunColor({ runUuid, colorValue })}
          />
        )}
        <Link
          componentId="mlflow.experiment_tracking.runs_table.run_name_link"
          to={Routes.getRunPageRoute(experimentId, runUuid)}
          css={styles.runLink}
          tabIndex={0}
          title={meta ? `${meta.modelTitle} — ${meta.epochsLabel}` : runName}
        >
          <span css={styles.runName}>{meta ? meta.modelTitle : runName}</span>
        </Link>
        {meta && (
          <div css={{ display: 'inline-flex', alignItems: 'center', gap: 4, flexShrink: 0 }}>
            <span
              title={meta.evalBadgeLabel}
              css={{
                padding: '0px 5px',
                borderRadius: 3,
                fontSize: 10,
                fontWeight: 600,
                lineHeight: '16px',
                backgroundColor: meta.evalKind === 'base' ? theme.colors.backgroundSecondary : '#ede9fe',
                color: meta.evalKind === 'base' ? theme.colors.textSecondary : '#4c1d95',
              }}
            >
              {meta.shortEpochLabel}
            </span>
            {meta.trainBadgeLabel &&
              (meta.trainRunHref ? (
                <a
                  href={meta.trainRunHref}
                  onClick={(e) => e.stopPropagation()}
                  title={`${meta.trainBadgeLabel} — click to open training run`}
                  css={{
                    padding: '0px 5px',
                    borderRadius: 3,
                    fontSize: 10,
                    fontWeight: 600,
                    lineHeight: '16px',
                    backgroundColor: trainBadgeColors.bg,
                    color: trainBadgeColors.fg,
                    textDecoration: 'none',
                    '&:hover': { textDecoration: 'underline' },
                  }}
                >
                  {meta.trainBadgeLabel.replace(/^Train:\s*/, 'train ')} ↗
                </a>
              ) : (
                <span
                  title={meta.trainBadgeLabel}
                  css={{
                    padding: '0px 5px',
                    borderRadius: 3,
                    fontSize: 10,
                    fontWeight: 600,
                    lineHeight: '16px',
                    backgroundColor: trainBadgeColors.bg,
                    color: trainBadgeColors.fg,
                  }}
                >
                  {meta.trainBadgeLabel.replace(/^Train:\s*/, 'train ')}
                </span>
              ))}
            {meta.evalKind !== 'base' && (
              <StageJudgementBadgeControl
                trial={meta.modelTitle}
                runUuid={runUuid}
                tags={tags}
                currentCheckpoint={tags?.['checkpoint']?.value}
                compact
              />
            )}
          </div>
        )}
      </div>
    </div>
  );
});

const styles = {
  link: (theme: Theme) => ({
    display: 'inline-block',
    minWidth: theme.typography.fontSizeBase,
    minHeight: theme.typography.fontSizeBase,
  }),
  cellWrapper: {
    display: 'flex',
  },
  expanderButton: {
    svg: {
      width: 12,
      height: 12,
    },
  },
  runLink: {
    overflow: 'hidden',
    display: 'flex',
    gap: 8,
    alignItems: 'center',
    '&:focus-visible': {
      textDecoration: 'underline',
    },
  },
  runName: {
    overflow: 'hidden',
    textOverflow: 'ellipsis',
  },
  expanderWrapper: {
    display: 'none',
    '.ag-grid-expanders-visible &': {
      display: 'block',
    },
  },
  nestLevel: (theme: Theme) => ({
    display: 'flex',
    justifyContent: 'flex-end',
    height: theme.spacing.lg,
  }),
};
