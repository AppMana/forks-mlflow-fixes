import React, { useEffect, useState } from 'react';
import { Header, HomeIcon, TableSkeleton, TitleSkeleton, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { ScrollablePageWrapper } from '../common/components/ScrollablePageWrapper';
import { useQuery } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { useSearchParams } from '../common/utils/RoutingUtils';
import { ErrorView } from '../common/components/ErrorView';
import { ErrorCodes } from '../common/constants';
import { ErrorWrapper } from '../common/utils/ErrorWrapper';
import { MlflowService } from '../experiment-tracking/sdk/MlflowService';
import type { SearchExperimentsApiResponse } from '../experiment-tracking/types';
import { CreateExperimentModal } from '../experiment-tracking/components/modals/CreateExperimentModal';
import { useInvalidateExperimentList } from '../experiment-tracking/components/experiment-page/hooks/useExperimentListQuery';
import { FeaturesSection } from './components/features';
import {
  extractWorkspaceFromSearchParams,
  setActiveWorkspace,
  setLastUsedWorkspace,
} from '../workspaces/utils/WorkspaceUtils';
// Loaders and lazy imports for expensive components
import LogTracesDrawerLoader from './components/LogTracesDrawerLoader';
import { TelemetryInfoAlert } from '../telemetry/TelemetryInfoAlert';

const ExperimentsHomeView = React.lazy(() => import('./components/ExperimentsHomeView'));

type ExperimentQueryKey = ['home', 'recent-experiments'];

const RECENT_EXPERIMENTS_QUERY_KEY: ExperimentQueryKey = ['home', 'recent-experiments'];

const HIDDEN_BY_DEFAULT_EXPERIMENTS = new Set([
  'Default',
  'test-resnet-experiment',
  'MLflow Quickstart',
  'spellsource-gsvb-training',
  'spellsource-curriculum-test',
  'spellsource-ui-probe',
  'spellsource-caption-ab',
  'spellsource-qwen21-assistant-validation',
]);

const SPELLSOURCE_HUB_LINKS = [
  { id: '110', title: 'Qwen-Image 2.1', sub: 'T2I 100-ep (live) + Photoreal/Sil Edit', live: true },
  { id: '114', title: 'Flux.2 Klein 4B', sub: 'Photoreal Edit (live) + s8 T2I/Sil', live: true },
  { id: '157', title: 'Fibo 2.0', sub: 's16 T2I (live) + Photoreal/BBox/Sil Edit', live: true },
  { id: '111', title: 'Krea 2 14B', sub: '100-ep s8 T2I + Photoreal/BBox/Sil Edit', live: false },
  { id: '118', title: 'Iris 3B', sub: '100-ep s16-c1024 T2I', live: false },
  { id: '113', title: 'Ideogram 4', sub: '100-ep s4-c512 T2I', live: false },
  { id: '122', title: 'PixelDiT 1.3B', sub: 's16-c1024 T2I', live: false },
  { id: '123', title: 'MOSS-Image 5B', sub: 's16-c1024 T2I', live: false },
  { id: '120', title: 'VAE Decoder', sub: 'Pixel-art VAE fine-tuning', live: false },
];

const HomePage = () => {
  const { theme } = useDesignSystemTheme();
  const invalidateExperiments = useInvalidateExperimentList();
  const [isCreateModalOpen, setIsCreateModalOpen] = useState(false);
  const [showAllExperiments, setShowAllExperiments] = useState(false);
  const [searchParams] = useSearchParams();
  const workspaceFromQuery = extractWorkspaceFromSearchParams(searchParams);

  const { data, error, isLoading, refetch } = useQuery<
    SearchExperimentsApiResponse,
    Error,
    SearchExperimentsApiResponse,
    ExperimentQueryKey
  >(RECENT_EXPERIMENTS_QUERY_KEY, {
    queryFn: () =>
      MlflowService.searchExperiments([
        ['max_results', '30'],
        ['order_by', 'last_update_time DESC'],
      ]),
    staleTime: 30_000,
  });

  const allExperiments = data?.experiments;
  const experiments = React.useMemo(() => {
    if (!allExperiments) return undefined;
    if (showAllExperiments) return allExperiments;
    const filtered = allExperiments.filter(
      (e) => e.name.startsWith('spellsource-') && !HIDDEN_BY_DEFAULT_EXPERIMENTS.has(e.name),
    );
    return filtered.length > 0 ? filtered : allExperiments;
  }, [allExperiments, showAllExperiments]);
  const errorMessage = error instanceof ErrorWrapper ? error.getMessageField() : null;
  const isWorkspaceNotFoundError =
    workspaceFromQuery &&
    error instanceof ErrorWrapper &&
    error.getErrorCode() === ErrorCodes.RESOURCE_DOES_NOT_EXIST &&
    typeof errorMessage === 'string' &&
    errorMessage.includes(`'${workspaceFromQuery}'`) &&
    /^Workspace '.+' not found(?:\b|$)/.test(errorMessage);

  useEffect(() => {
    if (!isWorkspaceNotFoundError) {
      return;
    }

    setActiveWorkspace(null);
    setLastUsedWorkspace(null);
  }, [isWorkspaceNotFoundError]);

  const handleOpenCreateModal = () => setIsCreateModalOpen(true);
  const handleCloseCreateModal = () => setIsCreateModalOpen(false);
  const handleExperimentCreated = () => {
    handleCloseCreateModal();
    invalidateExperiments();
    refetch();
  };

  if (isWorkspaceNotFoundError) {
    return (
      <ErrorView
        statusCode={404}
        subMessage={`Workspace "${workspaceFromQuery}" was not found`}
        fallbackHomePageReactRoute="/"
        disableWorkspacePrefixOnFallback
      />
    );
  }

  return (
    <ScrollablePageWrapper
      css={{
        padding: theme.spacing.md,
        display: 'flex',
        flexDirection: 'column',
        gap: theme.spacing.md,
        height: 'min-content',
      }}
    >
      <Header
        title={
          <span css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
            <span
              css={{
                display: 'flex',
                borderRadius: theme.borders.borderRadiusSm,
                backgroundColor: theme.colors.backgroundSecondary,
                padding: theme.spacing.sm,
              }}
            >
              <HomeIcon />
            </span>
            <FormattedMessage defaultMessage="Welcome to MLflow" description="Home page hero title" />
          </span>
        }
      />
      <TelemetryInfoAlert />
      {/* Spellsource Diffusion Model Hub */}
      <div
        css={{
          display: 'flex',
          flexDirection: 'column',
          gap: 10,
          padding: '14px 16px',
          borderRadius: 8,
          backgroundColor: theme.colors.backgroundSecondary,
          border: `1px solid ${theme.colors.border}`,
        }}
      >
        <div
          css={{
            display: 'flex',
            flexWrap: 'wrap',
            alignItems: 'center',
            justifyContent: 'space-between',
            gap: 12,
          }}
        >
          <div>
            <div css={{ fontSize: 15, fontWeight: 700, color: theme.colors.textPrimary }}>
              Spellsource Diffusion Model Hub
            </div>
            <div css={{ fontSize: 12, color: theme.colors.textSecondary }}>
              Direct links to the Master Comparison Gallery (showing both Eval Checkpoint Epoch & Current Training
              Progress) and active Family Training Experiments
            </div>
          </div>
          <div css={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <a
              href="#/experiments/115?viewStateShareKey=final-eval-gallery&compareRunsMode=CHART"
              css={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: 6,
                padding: '7px 14px',
                borderRadius: 6,
                fontSize: 13,
                fontWeight: 700,
                backgroundColor: theme.colors.actionPrimaryBackgroundDefault,
                color: '#ffffff',
                textDecoration: 'none',
                '&:hover': { filter: 'brightness(1.06)' },
              }}
            >
              Open Master Model Comparison Gallery (Exp 115) ↗
            </a>
            <button
              type="button"
              onClick={() => setShowAllExperiments((v) => !v)}
              css={{
                padding: '6px 10px',
                borderRadius: 6,
                fontSize: 12,
                fontWeight: 600,
                cursor: 'pointer',
                backgroundColor: theme.colors.backgroundPrimary,
                color: theme.colors.textPrimary,
                border: `1px solid ${theme.colors.border}`,
              }}
            >
              {showAllExperiments
                ? `Showing All Experiments (${allExperiments?.length ?? 0}) — Click for Live Only`
                : `Showing Live & Interesting (${experiments?.length ?? 0}) — Click to Show All (${allExperiments?.length ?? 0})`}
            </button>
          </div>
        </div>
        <div css={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
          {SPELLSOURCE_HUB_LINKS.map((item) => (
            <a
              key={item.id}
              href={`#/experiments/${item.id}`}
              css={{
                display: 'flex',
                flexDirection: 'column',
                gap: 2,
                padding: '6px 10px',
                borderRadius: 6,
                textDecoration: 'none',
                backgroundColor: item.live ? '#dbeafe' : theme.colors.backgroundPrimary,
                border: `1px solid ${item.live ? '#93c5fd' : theme.colors.border}`,
                '&:hover': { borderColor: theme.colors.actionPrimaryBackgroundDefault },
              }}
            >
              <div css={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                <span css={{ fontSize: 12, fontWeight: 700, color: item.live ? '#1e40af' : theme.colors.textPrimary }}>
                  {item.title} (Exp {item.id}) ↗
                </span>
                {item.live && (
                  <span
                    css={{
                      padding: '0px 5px',
                      borderRadius: 3,
                      fontSize: 10,
                      fontWeight: 700,
                      backgroundColor: '#1d4ed8',
                      color: '#ffffff',
                    }}
                  >
                    LIVE
                  </span>
                )}
              </div>
              <span css={{ fontSize: 11, color: item.live ? '#1e3a8a' : theme.colors.textSecondary }}>{item.sub}</span>
            </a>
          ))}
        </div>
      </div>
      <React.Suspense fallback={<HomePageSectionSkeleton />}>
        <ExperimentsHomeView
          experiments={experiments}
          isLoading={isLoading}
          error={error}
          onCreateExperiment={handleOpenCreateModal}
          onRetry={refetch}
          onTagsUpdated={refetch}
        />
      </React.Suspense>

      <CreateExperimentModal
        isOpen={isCreateModalOpen}
        onClose={handleCloseCreateModal}
        onExperimentCreated={handleExperimentCreated}
      />
      <LogTracesDrawerLoader />
    </ScrollablePageWrapper>
  );
};

const HomePageSectionSkeleton = () => (
  <div>
    <TitleSkeleton />
    <TableSkeleton lines={3} />
  </div>
);

export default HomePage;
