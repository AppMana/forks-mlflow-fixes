import React, { useEffect, useMemo, useState } from 'react';
import {
  Button,
  DatabaseIcon,
  Spinner,
  Tag,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import type { KeyValueEntity } from '@mlflow/mlflow/src/common/types';
import type { RunEntity } from '@mlflow/mlflow/src/experiment-tracking/types';
import { MLFLOW_LOGGED_ARTIFACTS_TAG, LOG_IMAGE_TAG_INDICATOR, RunPageTabName } from '@mlflow/mlflow/src/experiment-tracking/constants';
import { MlflowService } from '@mlflow/mlflow/src/experiment-tracking/sdk/MlflowService';
import { RunViewDatasetsTab } from '@mlflow/mlflow/src/experiment-tracking/components/run-page/RunViewDatasetsTab';
import { Link } from '@mlflow/mlflow/src/common/utils/RoutingUtils';
import Routes from '@mlflow/mlflow/src/experiment-tracking/routes';
import { parseJSONSafe } from '@mlflow/mlflow/src/common/utils/TagUtils';

export interface ExperimentRunDatasetsSectionProps {
  experimentId: string;
  onHasRunDatasetsChange?: (hasDatasets: boolean, isLoading: boolean) => void;
}

const tagsArrayToRecord = (tags?: KeyValueEntity[]): Record<string, KeyValueEntity> => {
  const out: Record<string, KeyValueEntity> = {};
  if (!tags) return out;
  for (const t of tags) {
    if (t && t.key) {
      out[t.key] = t;
    }
  }
  return out;
};

export const ExperimentRunDatasetsSection = ({
  experimentId,
  onHasRunDatasetsChange,
}: ExperimentRunDatasetsSectionProps) => {
  const { theme } = useDesignSystemTheme();
  const [runs, setRuns] = useState<RunEntity[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedRunUuid, setSelectedRunUuid] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);

    MlflowService.searchRuns({
      experiment_ids: [experimentId],
      max_results: 50,
    })
      .then((res) => {
        if (cancelled) return;
        const allRuns: RunEntity[] = res?.runs ?? [];
        const datasetRuns = allRuns.filter((r) => {
          const hasInputs = (r.inputs?.datasetInputs?.length ?? 0) > 0;
          const tags = r.data?.tags ?? [];
          const hasLoggedTables = tags.some((t) => t.key === MLFLOW_LOGGED_ARTIFACTS_TAG);
          const hasLoggedImages = tags.some((t) => t.key === LOG_IMAGE_TAG_INDICATOR);
          return hasInputs || hasLoggedTables || hasLoggedImages;
        });

        const finalRuns = datasetRuns.length > 0 ? datasetRuns : allRuns.slice(0, 10);
        setRuns(finalRuns);
        if (finalRuns.length > 0) {
          setSelectedRunUuid(finalRuns[0].info.runUuid);
        }
        setLoading(false);
        onHasRunDatasetsChange?.(finalRuns.length > 0, false);
      })
      .catch(() => {
        if (!cancelled) {
          setLoading(false);
          onHasRunDatasetsChange?.(false, false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [experimentId, onHasRunDatasetsChange]);

  const selectedRun = useMemo(
    () => runs.find((r) => r.info.runUuid === selectedRunUuid) ?? runs[0],
    [runs, selectedRunUuid],
  );

  if (loading) {
    return (
      <div css={{ display: 'flex', alignItems: 'center', gap: 8, padding: theme.spacing.md }}>
        <Spinner size="small" />
        <Typography.Text color="secondary">Loading run-logged datasets and manifests…</Typography.Text>
      </div>
    );
  }

  if (runs.length === 0 || !selectedRun) {
    return null;
  }

  const selectedRunTags = tagsArrayToRecord(selectedRun.data?.tags);
  const selectedRunDatasets = selectedRun.inputs?.datasetInputs ?? [];

  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
        gap: theme.spacing.md,
        paddingRight: theme.spacing.md,
      }}
    >
      {/* Run-Logged Dataset Selector Bar */}
      <div
        css={{
          border: `1px solid ${theme.colors.border}`,
          borderRadius: theme.borders.borderRadiusMd,
          backgroundColor: theme.colors.backgroundSecondary,
          padding: theme.spacing.md,
          display: 'flex',
          flexDirection: 'column',
          gap: theme.spacing.sm,
        }}
      >
        <div css={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', justifyContent: 'space-between', gap: 8 }}>
          <div css={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <DatabaseIcon css={{ color: theme.colors.actionPrimaryBackgroundDefault }} />
            <Typography.Title level={4} withoutMargins>
              Run-Logged Datasets, Tables & Visual Galleries ({runs.length} {runs.length === 1 ? 'run' : 'runs'})
            </Typography.Title>
          </div>
          <Link
            componentId="mlflow.experiment-datasets.open-run-datasets-tab"
            to={Routes.getRunPageTabRoute(experimentId, selectedRun.info.runUuid, RunPageTabName.DATASETS)}
            css={{ fontWeight: 600, fontSize: 13 }}
          >
            Open Dedicated Run Datasets Page ({selectedRun.info.runName || selectedRun.info.runUuid.slice(0, 8)}) ↗
          </Link>
        </div>

        <div css={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
          {runs.map((run) => {
            const isSelected = run.info.runUuid === selectedRun.info.runUuid;
            const dsInputs = run.inputs?.datasetInputs ?? [];
            const firstDs = dsInputs[0]?.dataset;
            const profileObj = firstDs?.profile ? parseJSONSafe(firstDs.profile) : null;
            const numRows = profileObj?.num_rows;
            return (
              <Button
                key={run.info.runUuid}
                componentId="mlflow.experiment-datasets.select-run"
                size="small"
                type={isSelected ? 'primary' : 'tertiary'}
                onClick={() => setSelectedRunUuid(run.info.runUuid)}
              >
                <span css={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
                  <strong>{run.info.runName || run.info.runUuid.slice(0, 8)}</strong>
                  {firstDs && (
                    <Tag componentId="mlflow.experiment-datasets.run-ds-tag" color="indigo">
                      {firstDs.name}
                      {numRows !== undefined ? ` (${numRows} rows)` : ''}
                    </Tag>
                  )}
                </span>
              </Button>
            );
          })}
        </div>
      </div>

      {/* Render full RunViewDatasetsTab for the selected run */}
      <RunViewDatasetsTab
        key={selectedRun.info.runUuid}
        runUuid={selectedRun.info.runUuid}
        experimentId={experimentId}
        runInfo={selectedRun.info}
        tags={selectedRunTags}
        datasets={selectedRunDatasets}
        artifactUri={selectedRun.info.artifactUri ?? undefined}
      />
    </div>
  );
};
