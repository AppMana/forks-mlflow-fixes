import React, { useEffect, useMemo, useState } from 'react';
import {
  Button,
  DatabaseIcon,
  Empty,
  ImageIcon,
  Spinner,
  TableIcon,
  Tag,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import type { KeyValueEntity } from '../../../common/types';
import type { RunDatasetWithTags, RunInfoEntity } from '../../types';
import type { UseGetRunQueryResponseRunInfo } from './hooks/useGetRunQuery';
import { MLFLOW_LOGGED_ARTIFACTS_TAG, MLFLOW_RUN_DATASET_CONTEXT_TAG } from '../../constants';
import { parseJSONSafe } from '../../../common/utils/TagUtils';
import { getArtifactLocationUrl } from '../../../common/utils/ArtifactUtils';
import { MlflowService } from '../../sdk/MlflowService';
import { ShowArtifactLoggedTableView } from '../artifact-view-components/ShowArtifactLoggedTableView';
import { ExperimentViewDatasetSchema } from '../experiment-page/components/runs/ExperimentViewDatasetSchema';
import { ExperimentViewDatasetDigest } from '../experiment-page/components/runs/ExperimentViewDatasetDigest';
import { ExperimentViewDatasetSourceType } from '../experiment-page/components/runs/ExperimentViewDatasetSourceType';
import { ExperimentViewDatasetSourceURL } from '../experiment-page/components/runs/ExperimentViewDatasetSourceURL';
import { Link } from '../../../common/utils/RoutingUtils';
import Routes from '../../routes';

interface ArtifactFileEntry {
  path: string;
  is_dir: boolean;
  file_size?: number | string;
}

const formatBytes = (bytes?: number | string): string => {
  const num = typeof bytes === 'string' ? parseInt(bytes, 10) : bytes;
  if (num === undefined || Number.isNaN(num)) return '';
  if (num < 1024) return `${num} B`;
  if (num < 1024 * 1024) return `${(num / 1024).toFixed(1)} KB`;
  return `${(num / (1024 * 1024)).toFixed(2)} MB`;
};

const formatImageTitle = (path: string): string => {
  const filename = path.split('/').pop() || path;
  const withoutStep = filename.replace(/%step%\d+%timestamp%\d+%.*$/, '');
  const withoutExt = withoutStep.replace(/\.(png|jpg|jpeg|webp|gif)$/i, '');
  return withoutExt.replace(/_/g, ' ');
};

export interface RunViewDatasetsTabProps {
  runUuid: string;
  experimentId: string;
  runInfo: RunInfoEntity | UseGetRunQueryResponseRunInfo;
  tags: Record<string, KeyValueEntity>;
  datasets?: RunDatasetWithTags[];
  artifactUri?: string;
}

export const RunViewDatasetsTab = ({
  runUuid,
  experimentId,
  tags,
  datasets = [],
  artifactUri,
}: RunViewDatasetsTabProps) => {
  const { theme } = useDesignSystemTheme();
  const [selectedDatasetIdx, setSelectedDatasetIdx] = useState(0);
  const [selectedTablePath, setSelectedTablePath] = useState<string | null>(null);
  const [rootFiles, setRootFiles] = useState<ArtifactFileEntry[]>([]);
  const [imageFiles, setImageFiles] = useState<ArtifactFileEntry[]>([]);
  const [manifestFiles, setManifestFiles] = useState<ArtifactFileEntry[]>([]);
  const [loadingArtifacts, setLoadingArtifacts] = useState(true);
  const [lightboxUrl, setLightboxUrl] = useState<string | null>(null);
  const [lightboxTitle, setLightboxTitle] = useState<string>('');
  const [showSchema, setShowSchema] = useState(true);

  useEffect(() => {
    let cancelled = false;
    setLoadingArtifacts(true);

    MlflowService.listArtifacts({ run_uuid: runUuid })
      .then(async (res: { files?: ArtifactFileEntry[] }) => {
        if (cancelled) return;
        const files = res?.files ?? [];
        setRootFiles(files);

        const hasImagesDir = files.some((f) => f.is_dir && f.path === 'images');
        const hasManifestsDir = files.some((f) => f.is_dir && f.path === 'manifests');

        const [imgRes, manRes] = await Promise.all([
          hasImagesDir
            ? MlflowService.listArtifacts({ run_uuid: runUuid, path: 'images' }).catch(() => ({ files: [] }))
            : Promise.resolve({ files: [] }),
          hasManifestsDir
            ? MlflowService.listArtifacts({ run_uuid: runUuid, path: 'manifests' }).catch(() => ({ files: [] }))
            : Promise.resolve({ files: [] }),
        ]);

        if (cancelled) return;
        setImageFiles(imgRes?.files ?? []);
        setManifestFiles(manRes?.files ?? []);
        setLoadingArtifacts(false);
      })
      .catch(() => {
        if (!cancelled) {
          setLoadingArtifacts(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [runUuid]);

  const loggedTablePaths = useMemo(() => {
    const paths = new Set<string>();
    const rawTag = tags?.[MLFLOW_LOGGED_ARTIFACTS_TAG]?.value;
    if (rawTag) {
      const parsed = parseJSONSafe(rawTag);
      if (Array.isArray(parsed)) {
        for (const item of parsed) {
          if (item && item.type === 'table' && typeof item.path === 'string') {
            paths.add(item.path);
          }
        }
      }
    }
    for (const f of rootFiles) {
      if (!f.is_dir && f.path.endsWith('.json')) {
        paths.add(f.path);
      }
    }
    return Array.from(paths);
  }, [tags, rootFiles]);

  useEffect(() => {
    if (loggedTablePaths.length > 0 && (!selectedTablePath || !loggedTablePaths.includes(selectedTablePath))) {
      const preferred =
        loggedTablePaths.find((p) => p.includes('extracted_items')) ||
        loggedTablePaths.find((p) => p.includes('unified_corpus_summary')) ||
        loggedTablePaths[0];
      setSelectedTablePath(preferred);
    }
  }, [loggedTablePaths, selectedTablePath]);

  const galleryImages = useMemo(() => {
    const nonDir = imageFiles.filter(
      (f) => !f.is_dir && /\.(png|jpg|jpeg|webp|gif)$/i.test(f.path) && !f.path.includes('.Comparable.'),
    );
    return nonDir.sort((a, b) => a.path.localeCompare(b.path));
  }, [imageFiles]);

  const activeDataset = datasets[selectedDatasetIdx] ?? datasets[0];
  const parsedProfile = useMemo(() => {
    if (!activeDataset?.dataset?.profile || activeDataset.dataset.profile === 'null') {
      return null;
    }
    return parseJSONSafe(activeDataset.dataset.profile);
  }, [activeDataset]);

  const hasAnyContent =
    datasets.length > 0 || loggedTablePaths.length > 0 || galleryImages.length > 0 || manifestFiles.length > 0;

  if (!loadingArtifacts && !hasAnyContent) {
    return (
      <div css={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', padding: theme.spacing.lg }}>
        <Empty
          image={<DatabaseIcon />}
          title={
            <FormattedMessage
              defaultMessage="No datasets logged for this run"
              description="Run page > Datasets tab > empty state title"
            />
          }
          description={
            <FormattedMessage
              defaultMessage="Log datasets using mlflow.log_input(), mlflow.log_table(), or mlflow.log_image() to inspect schemas, interactive tables, and visual contact sheets here."
              description="Run page > Datasets tab > empty state description"
            />
          }
        />
      </div>
    );
  }

  return (
    <div
      css={{
        flex: 1,
         display: 'flex',
        flexDirection: 'column',
        gap: theme.spacing.md,
        paddingBottom: theme.spacing.lg,
        overflowY: 'auto',
      }}
    >
      {/* 1. Logged Dataset Inputs (mlflow.log_input) */}
      {datasets.length > 0 && activeDataset && (
        <div
          css={{
            border: `1px solid ${theme.colors.border}`,
            borderRadius: theme.borders.borderRadiusMd,
            backgroundColor: theme.colors.backgroundPrimary,
            padding: theme.spacing.md,
            display: 'flex',
            flexDirection: 'column',
            gap: theme.spacing.md,
          }}
        >
          <div css={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', justifyContent: 'space-between', gap: 8 }}>
            <div css={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
              <DatabaseIcon css={{ color: theme.colors.actionPrimaryBackgroundDefault }} />
              <Typography.Title level={3} withoutMargins>
                {activeDataset.dataset.name}
              </Typography.Title>
              {activeDataset.tags
                ?.filter((t) => t.key === MLFLOW_RUN_DATASET_CONTEXT_TAG)
                .map((t) => (
                  <Tag key={t.key} componentId="mlflow.run-datasets-tab.context-tag" color="indigo">
                    {t.value}
                  </Tag>
                ))}
              <Tag componentId="mlflow.run-datasets-tab.digest-tag">digest: {activeDataset.dataset.digest}</Tag>
            </div>
            {datasets.length > 1 && (
              <div css={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                {datasets.map((ds, idx) => (
                  <Button
                    key={`${ds.dataset.name}-${ds.dataset.digest}`}
                    componentId="mlflow.run-datasets-tab.dataset-switch"
                    size="small"
                    type={idx === selectedDatasetIdx ? 'primary' : 'tertiary'}
                    onClick={() => setSelectedDatasetIdx(idx)}
                  >
                    {ds.dataset.name} ({ds.dataset.digest})
                  </Button>
                ))}
              </div>
            )}
          </div>

          <div css={{ display: 'flex', flexWrap: 'wrap', gap: theme.spacing.lg, alignItems: 'center' }}>
            <ExperimentViewDatasetDigest datasetWithTags={activeDataset} />
            <ExperimentViewDatasetSourceType datasetWithTags={activeDataset} />
            <ExperimentViewDatasetSourceURL datasetWithTags={activeDataset} />
          </div>

          {/* Structured Dataset Profile Breakdown */}
          {parsedProfile && typeof parsedProfile === 'object' && (
            <div
              css={{
                display: 'flex',
                flexDirection: 'column',
                gap: theme.spacing.sm,
                padding: theme.spacing.sm,
                borderRadius: theme.borders.borderRadiusSm,
                backgroundColor: theme.colors.backgroundSecondary,
              }}
            >
              <Typography.Text bold>Dataset Profile & Breakdown</Typography.Text>
              <div css={{ display: 'flex', flexWrap: 'wrap', gap: theme.spacing.sm }}>
                {Object.entries(parsedProfile)
                  .filter(([, v]) => v === null || typeof v !== 'object')
                  .map(([k, v]) => (
                    <div
                      key={k}
                      css={{
                        padding: '6px 12px',
                        borderRadius: 6,
                        backgroundColor: theme.colors.backgroundPrimary,
                        border: `1px solid ${theme.colors.borderDecorative}`,
                        display: 'flex',
                        flexDirection: 'column',
                        gap: 2,
                      }}
                    >
                      <span css={{ fontSize: 11, color: theme.colors.textSecondary }}>{k.replace(/_/g, ' ')}</span>
                      <span css={{ fontSize: 13, fontWeight: 700, color: theme.colors.textPrimary }}>
                        {String(v)}
                      </span>
                    </div>
                  ))}
              </div>
              {Object.entries(parsedProfile)
                .filter(([, v]) => v !== null && typeof v === 'object' && !Array.isArray(v))
                .map(([groupKey, groupObj]) => (
                  <div key={groupKey} css={{ display: 'flex', flexDirection: 'column', gap: 4, marginTop: 4 }}>
                    <span css={{ fontSize: 12, fontWeight: 600, color: theme.colors.textSecondary }}>
                      {groupKey.replace(/_/g, ' ').toUpperCase()}
                    </span>
                    <div css={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                      {Object.entries(groupObj as Record<string, any>).map(([subKey, subVal]) => (
                        <span
                          key={subKey}
                          css={{
                            padding: '3px 8px',
                            borderRadius: 4,
                            fontSize: 12,
                            fontWeight: 600,
                            backgroundColor: theme.colors.backgroundPrimary,
                            border: `1px solid ${theme.colors.border}`,
                          }}
                        >
                          {subKey}: <strong>{String(subVal)}</strong>
                        </span>
                      ))}
                    </div>
                  </div>
                ))}
            </div>
          )}

          {/* Dataset Schema */}
          <div>
            <div css={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 6 }}>
              <Typography.Text bold>Dataset Column Schema</Typography.Text>
              <Button
                componentId="mlflow.run-datasets-tab.toggle-schema"
                size="small"
                type="tertiary"
                onClick={() => setShowSchema((s) => !s)}
              >
                {showSchema ? 'Hide Schema' : 'Show Schema'}
              </Button>
            </div>
            {showSchema && (
              <div css={{ maxHeight: 300, overflowY: 'auto' }}>
                <ExperimentViewDatasetSchema datasetWithTags={activeDataset} />
              </div>
            )}
          </div>
        </div>
      )}

      {/* 2. Parquet Manifests & Dataset Files */}
      {manifestFiles.length > 0 && (
        <div
          css={{
            border: `1px solid ${theme.colors.border}`,
            borderRadius: theme.borders.borderRadiusMd,
            backgroundColor: theme.colors.backgroundPrimary,
            padding: theme.spacing.md,
            display: 'flex',
            flexDirection: 'column',
            gap: theme.spacing.sm,
          }}
        >
          <Typography.Title level={4} withoutMargins>
            Parquet Manifests & Dataset Files ({manifestFiles.length})
          </Typography.Title>
          <div css={{ display: 'flex', flexWrap: 'wrap', gap: theme.spacing.sm }}>
            {manifestFiles.map((file) => (
              <div
                key={file.path}
                css={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 10,
                  padding: '8px 12px',
                  borderRadius: 6,
                  backgroundColor: theme.colors.backgroundSecondary,
                  border: `1px solid ${theme.colors.border}`,
                }}
              >
                <TableIcon />
                <div css={{ display: 'flex', flexDirection: 'column' }}>
                  <span css={{ fontWeight: 600, fontSize: 13 }}>{file.path}</span>
                  {file.file_size !== undefined && (
                    <span css={{ fontSize: 11, color: theme.colors.textSecondary }}>{formatBytes(file.file_size)}</span>
                  )}
                </div>
                <a
                  href={getArtifactLocationUrl(file.path, runUuid)}
                  download
                  css={{
                    padding: '4px 10px',
                    borderRadius: 4,
                    fontSize: 12,
                    fontWeight: 600,
                    backgroundColor: theme.colors.actionPrimaryBackgroundDefault,
                    color: '#fff',
                    textDecoration: 'none',
                  }}
                >
                  Download
                </a>
                <Link
                  componentId="mlflow.run-datasets-tab.manifest-artifact-link"
                  to={Routes.getRunPageTabRoute(experimentId, runUuid, `artifacts/${file.path}`)}
                  css={{ fontSize: 12 }}
                >
                  Inspect in Artifacts ↗
                </Link>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* 3. Interactive Logged Tables (mlflow.log_table) */}
      {loggedTablePaths.length > 0 && (
        <div
          css={{
            border: `1px solid ${theme.colors.border}`,
            borderRadius: theme.borders.borderRadiusMd,
            backgroundColor: theme.colors.backgroundPrimary,
            padding: theme.spacing.md,
            display: 'flex',
            flexDirection: 'column',
            gap: theme.spacing.sm,
          }}
        >
          <div css={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', justifyContent: 'space-between', gap: 8 }}>
            <div css={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <TableIcon css={{ color: theme.colors.actionPrimaryBackgroundDefault }} />
              <Typography.Title level={4} withoutMargins>
                Interactive Dataset Tables ({loggedTablePaths.length})
              </Typography.Title>
            </div>
            <div css={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
              {loggedTablePaths.map((tablePath) => (
                <Button
                  key={tablePath}
                  componentId="mlflow.run-datasets-tab.select-table"
                  size="small"
                  type={selectedTablePath === tablePath ? 'primary' : 'secondary'}
                  onClick={() => setSelectedTablePath(tablePath)}
                >
                  {tablePath}
                </Button>
              ))}
            </div>
          </div>

          {selectedTablePath && (
            <div
              css={{
                height: 640,
                border: `1px solid ${theme.colors.borderDecorative}`,
                borderRadius: theme.borders.borderRadiusSm,
                overflow: 'hidden',
                display: 'flex',
                flexDirection: 'column',
              }}
            >
              <ShowArtifactLoggedTableView
                runUuid={runUuid}
                path={selectedTablePath}
                artifactRootUri={artifactUri}
              />
            </div>
          )}
        </div>
      )}

      {/* 4. Visual Dataset Gallery (mlflow.log_image) */}
      {galleryImages.length > 0 && (
        <div
          css={{
            border: `1px solid ${theme.colors.border}`,
            borderRadius: theme.borders.borderRadiusMd,
            backgroundColor: theme.colors.backgroundPrimary,
            padding: theme.spacing.md,
            display: 'flex',
            flexDirection: 'column',
            gap: theme.spacing.md,
          }}
        >
          <div css={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <ImageIcon css={{ color: theme.colors.actionPrimaryBackgroundDefault }} />
            <Typography.Title level={4} withoutMargins>
              Dataset Contact Sheets & Asset Previews ({galleryImages.length})
            </Typography.Title>
            <Typography.Text color="secondary">
              Click any contact sheet or sprite strip to inspect at full pixel-art resolution
            </Typography.Text>
          </div>
          <div
            css={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fill, minmax(360px, 1fr))',
              gap: theme.spacing.md,
            }}
          >
            {galleryImages.map((img) => {
              const url = getArtifactLocationUrl(img.path, runUuid);
              const title = formatImageTitle(img.path);
              return (
                <div
                  key={img.path}
                  onClick={() => {
                    setLightboxUrl(url);
                    setLightboxTitle(title);
                  }}
                  css={{
                    cursor: 'pointer',
                    border: `1px solid ${theme.colors.border}`,
                    borderRadius: theme.borders.borderRadiusSm,
                    backgroundColor: theme.colors.backgroundSecondary,
                    overflow: 'hidden',
                    display: 'flex',
                    flexDirection: 'column',
                    '&:hover': {
                      borderColor: theme.colors.actionPrimaryBackgroundDefault,
                    },
                  }}
                >
                  <div
                    css={{
                      padding: '6px 10px',
                      borderBottom: `1px solid ${theme.colors.borderDecorative}`,
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'space-between',
                      backgroundColor: theme.colors.backgroundPrimary,
                    }}
                  >
                    <span css={{ fontWeight: 600, fontSize: 12 }}>{title}</span>
                    <span css={{ fontSize: 11, color: theme.colors.textSecondary }}>
                      {formatBytes(img.file_size)}
                    </span>
                  </div>
                  <div
                    css={{
                      padding: 8,
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      minHeight: 220,
                      maxHeight: 360,
                      backgroundImage:
                        'linear-gradient(45deg, rgba(128,128,128,0.16) 25%, transparent 25%), ' +
                        'linear-gradient(-45deg, rgba(128,128,128,0.16) 25%, transparent 25%), ' +
                        'linear-gradient(45deg, transparent 75%, rgba(128,128,128,0.16) 75%), ' +
                        'linear-gradient(-45deg, transparent 75%, rgba(128,128,128,0.16) 75%)',
                      backgroundSize: '16px 16px',
                      backgroundPosition: '0 0, 0 8px, 8px -8px, -8px 0',
                    }}
                  >
                    <img
                      src={url}
                      alt={title}
                      loading="lazy"
                      css={{
                        maxWidth: '100%',
                        maxHeight: 340,
                        objectFit: 'contain',
                        imageRendering: 'pixelated',
                      }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {loadingArtifacts && (
        <div css={{ display: 'flex', alignItems: 'center', gap: 8, padding: theme.spacing.md }}>
          <Spinner size="small" />
          <Typography.Text color="secondary">Loading dataset tables and visual previews…</Typography.Text>
        </div>
      )}

      {/* Fullscreen Pixel-Art Lightbox */}
      {lightboxUrl && (
        <div
          onClick={() => setLightboxUrl(null)}
          css={{
            position: 'fixed',
            inset: 0,
            zIndex: 9999,
            backgroundColor: 'rgba(0, 0, 0, 0.85)',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            padding: 24,
            cursor: 'zoom-out',
          }}
        >
          <div
            css={{
              color: '#fff',
              fontSize: 16,
              fontWeight: 700,
              marginBottom: 12,
              display: 'flex',
              alignItems: 'center',
              gap: 16,
            }}
          >
            <span>{lightboxTitle}</span>
            <a
              href={lightboxUrl}
              target="_blank"
              rel="noreferrer"
              onClick={(e) => e.stopPropagation()}
              css={{ color: '#93c5fd', fontSize: 13 }}
            >
              Open original PNG ↗
            </a>
          </div>
          <img
            src={lightboxUrl}
            alt={lightboxTitle}
            css={{
              maxWidth: '95vw',
              maxHeight: '85vh',
              objectFit: 'contain',
              imageRendering: 'pixelated',
              backgroundColor: '#1e1e24',
              borderRadius: 6,
              padding: 8,
            }}
          />
        </div>
      )}
    </div>
  );
};
