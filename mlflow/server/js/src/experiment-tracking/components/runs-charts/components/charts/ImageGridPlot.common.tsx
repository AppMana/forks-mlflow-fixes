import { ImageIcon, Spinner, Tooltip } from '@databricks/design-system';
import { useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { getArtifactLocationUrl } from '@mlflow/mlflow/src/common/utils/ArtifactUtils';
import type { ImageEntity } from '@mlflow/mlflow/src/experiment-tracking/types';
import type { KeyValueEntity } from '@mlflow/mlflow/src/common/types';
import { useState, useEffect } from 'react';
import { Typography } from '@databricks/design-system';
import { ImagePreviewGroup, Image } from '../../../../../shared/building_blocks/Image';
import { RunColorPill } from '@mlflow/mlflow/src/experiment-tracking/components/experiment-page/components/RunColorPill';

export const MIN_GRID_IMAGE_SIZE = 400;

// Params like 'name' and 'run_id' duplicate the run name so we skip them
const REDUNDANT_PARAM_KEYS = new Set(['name', 'run_name', 'run_id']);

export const formatRunParams = (params: Record<string, { key: string; value: string | number }>): string => {
  const entries = Object.values(params).filter((p) => !REDUNDANT_PARAM_KEYS.has(p.key));
  if (entries.length === 0) return '';
  return entries.map((p) => `${p.key}=${p.value}`).join(', ');
};

export interface PromptCatalogEntry {
  title: string;
  summary: string;
  fullText: string;
}

export const SPELLSOURCE_PROMPT_CATALOG: Record<string, PromptCatalogEntry> = {
  'dinky_cook': {
    title: 'Dinky Diner Cook (30×58 cells)',
    summary: 'Cheerful diner cook in tall white toque, white jacket, red neckerchief & white apron',
    fullText:
      "A cheerful diner cook, a round-faced middle-aged man with a thick black mustache and a big smile, stands front-on in full body with his arms at his sides, wearing a tall white chef's toque, a white double-breasted jacket, a small red neckerchief, a white apron, grey trousers and black shoes. Spellsource pixel art: large hard-edged square pixels, thirty across and fifty-eight high, a one-pixel purple-black outline in color #21215C, flat cel shading, light from the upper left.",
  },
  'dinky_pepperoni-pizza': {
    title: 'Pepperoni Pizza (44×40 cells)',
    summary: 'Round pepperoni pizza with golden-brown crust, melted yellow-orange cheese & 8 red slices',
    fullText:
      'A whole round pepperoni pizza seen from above at a slight tilt, a thick golden-brown crust ring around melted yellow-orange cheese topped with eight round red pepperoni slices. Spellsource pixel art: large hard-edged square pixels, forty-four across and forty high, a one-pixel purple-black outline in color #21215C around the silhouette and the inner edge of the crust, flat cel shading, light from the upper left.',
  },
  'dinky_spaghetti-meatballs': {
    title: 'Spaghetti & Meatballs (48×36 cells)',
    summary: 'White oval plate with pale blue rim, yellow noodles, red tomato sauce, 3 meatballs & basil',
    fullText:
      'A plate of spaghetti and meatballs, a white oval plate with a pale blue rim heaped with yellow noodles under red tomato sauce, topped with three round brown meatballs and a small sprig of green basil. Spellsource pixel art: large hard-edged square pixels, forty-eight across and thirty-six high, a one-pixel purple-black outline in color #21215C, flat cel shading in two tones per material, light from the upper left.',
  },
  'dinky_soda-cup': {
    title: 'Fountain Soda Cup (26×44 cells)',
    summary: 'Tall red paper cup with white wavy stripe band, domed icy-blue lid & striped straw',
    fullText:
      'A soda cup, a tall red paper cup with a white wavy stripe band around its middle, a domed pale icy blue lid and a red-and-white striped straw leaning out of the lid toward the upper right. Spellsource pixel art: large hard-edged square pixels, twenty-six across and forty-four high, a one-pixel purple-black outline in color #21215C, flat cel shading in two tones per material, light from the upper left.',
  },
  'dinky_fire-oven': {
    title: 'Brick Pizza Oven (46×44 cells)',
    summary: 'Igloo-shaped terracotta brick pizza oven with 3 glowing flames on a grey stone plinth',
    fullText:
      'A domed brick pizza oven, an igloo-shaped oven of orange-red bricks with a dark arched mouth glowing with three yellow and orange flames, sitting on a grey stone base. Spellsource pixel art: large hard-edged square pixels, forty-six across and forty-four high, a one-pixel purple-black outline in color #21215C, flat cel shading, light from the upper left.',
  },
  'dinky_alarm-clock': {
    title: 'Twin-Bell Alarm Clock (34×36 cells)',
    summary: 'Red twin-bell kitchen timer with off-white dial, silver bells & hammer, and 2 short legs',
    fullText:
      'A red twin-bell alarm clock used as a kitchen timer, a round red body around an off-white face with near-black hands and tick marks, two silver bells on top joined by a small hammer, and two short silver legs. Spellsource pixel art: large hard-edged square pixels, thirty-four across and thirty-six high, a one-pixel purple-black outline in color #21215C, flat cel shading in two tones per material, light from the upper left.',
  },
  'dinky_hot-dog': {
    title: 'Hot Dog with Mustard (46×26 cells)',
    summary: 'Golden-brown bun split around a reddish-brown sausage with a wavy yellow mustard zigzag',
    fullText:
      'A hot dog with mustard, a golden-brown bun split lengthwise around a reddish-brown sausage with a wavy yellow mustard line zigzagging along its top. Spellsource pixel art: large hard-edged square pixels, forty-six across and twenty-six high, a one-pixel purple-black outline in color #21215C, flat cel shading, light from the upper left.',
  },
  'validation_00': {
    title: 'Validation 00 — Horned Shaman (34×58 cells)',
    summary: 'Robed humanoid with spiked navy mane, curled horn flaps, pink diamond face & goggle eyes',
    fullText:
      'A thin robed humanoid stands frontal, crowned by a wild spiked dark navy mane and two large curled horn-like flaps sweeping out on either side of a small diamond-shaped pink face with round goggle eyes. Spellsource pixel art: large hard-edged square pixels, thirty-four across and fifty-eight high, a one-pixel purple-black outline in color #212160, flat two-step cel shading, light from the upper left.',
  },
  'validation_01': {
    title: 'Validation 01 — Wooden Box Spirit (37×33 cells)',
    summary: 'Open-topped wooden box spirit with checkered rim & bright blue snake tendril curling out',
    fullText:
      'A wooden box spirit seen from a high three-quarter angle, its open top revealing a dark purple interior ringed with a tan checkered rim, a bright blue snake-like tendril curling out of an arched hole in its front. Spellsource pixel art: large hard-edged square pixels, thirty-seven across and thirty-three high, a one-pixel purple-black outline in color #212160, flat shading, light from the upper left.',
  },
  'validation_02': {
    title: 'Validation 02 — Blue Oni Goblin (32×40 cells)',
    summary: 'Small sky-blue Oni goblin with pointed hood-like head, long ears, fanged grin & wooden club',
    fullText:
      'A small blue-skinned Oni goblin with a pointed hood-like head, long ears and a fanged grin stands gripping a knobbed wooden club held upright at its side. Spellsource pixel art: large hard-edged square pixels, thirty-two across and forty high, a one-pixel purple-black outline in color #1D1E5C, flat two-tone cel shading, light from the upper left.',
  },
  'validation_03': {
    title: 'Validation 03 — Dragovian Enchanter (37×57 cells)',
    summary: 'Robed Dragovian enchanter raising 4 clawed hands in turquoise magic aura with floating glyphs',
    fullText:
      'A robed Dragovian enchanter raises four clawed hands in a casting pose, wrapped in a crackling aura of turquoise Lun motes, a row of rune-like glyphs floating above its head. Spellsource pixel art: large hard-edged square pixels, thirty-seven across and fifty-seven high, a one-pixel purple-black outline in color #21215C, flat colour fills, light from the upper left.',
  },
  'generic_00': {
    title: 'Generic 00 — Moon Carriage',
    summary: 'Horses pulling a carriage on the moon with Statue of Liberty, Great Pyramid & Earth in sky',
    fullText:
      "Horses pulling a carriage on the moon's surface, with the Statue of Liberty and Great Pyramid in the background. The Planet Earth can be seen in the sky.",
  },
  'generic_01': {
    title: 'Generic 01 — White Towel',
    summary: 'A white towel (generalisation monitor)',
    fullText: 'a white towel',
  },
  'generic_02': {
    title: 'Generic 02 — COFFEE Burger',
    summary: 'Burger patty with bottom bun, lettuce & tomatoes, "COFFEE" written in mustard',
    fullText: 'A burger patty, with the bottom bun and lettuce and tomatoes. "COFFEE" written on it in mustard',
  },
  'generic_03': {
    title: 'Generic 03 — Pumpkin',
    summary: 'A pumpkin (generalisation monitor)',
    fullText: 'a pumpkin',
  },
  'generic_04': {
    title: 'Generic 04 — Chimpanzee at Piano',
    summary: 'A chimpanzee wearing a bowtie and playing a piano',
    fullText: 'a chimpanzee wearing a bowtie and playing a piano',
  },
  'generic_05': {
    title: 'Generic 05 — Tennis Court',
    summary: 'A tennis court (generalisation monitor)',
    fullText: 'a tennis court',
  },
  'generic_06': {
    title: 'Generic 06 — Four Dragons & Dinosaur',
    summary: 'Four dragons surrounding a dinosaur',
    fullText: 'Four dragons surrounding a dinosaur',
  },
  'generic_07': {
    title: 'Generic 07 — OOPS Wine Spill',
    summary: 'Glass of red wine tipped over on a couch with a stain writing "OOPS"',
    fullText: 'A glass of red wine tipped over on a couch, with a stain that writes "OOPS" on the couch.',
  },
  'generic_08': {
    title: 'Generic 08 — Cat Dreaming of Tiger',
    summary: 'A cat dreaming about becoming a tiger',
    fullText: 'A cat dreaming about becoming a tiger',
  },
  'generic_09': {
    title: 'Generic 09 — Eye of Jupiter',
    summary: 'A close-up of the eye of the planet Jupiter',
    fullText: 'a close-up of the eye of the planet Jupiter',
  },
  'generic_10': {
    title: 'Generic 10 — Comic Book Supervillain',
    summary: 'A comic book supervillain',
    fullText: 'a comic book supervillian',
  },
  'generic_11': {
    title: 'Generic 11 — Friendly Car',
    summary: 'A friendly car',
    fullText: 'a friendly car',
  },
  'generic_12': {
    title: 'Generic 12 — Laptop with Photographs',
    summary: 'A laptop screen showing a bunch of photographs',
    fullText: 'a laptop screen showing a bunch of photographs',
  },
  'generic_13': {
    title: 'Generic 13 — Armadillo Playing Bagpipe',
    summary: 'High-quality photograph of an armadillo playing a bagpipe on one leg',
    fullText: 'a high-quality photograph of an armadillo playing a bagpipe while standing on one leg',
  },
  'generic_14': {
    title: 'Generic 14 — Stone Path',
    summary: 'A stone path',
    fullText: 'a stone path',
  },
  'generic_15': {
    title: 'Generic 15 — Yellow Square on Black',
    summary: 'A black background with a large yellow square',
    fullText: 'a black background with a large yellow square',
  },
};

export interface ParsedImageKey {
  rawKey: string;
  promptName: string;
  stem: string;
  canvasSize: string | null;
  seed: string | null;
  arm: string;
  stageNumber: number;
  stageLabel: string;
  snappedStatus: string;
  isSnapped: boolean;
  promptInfo: PromptCatalogEntry | null;
}

export const parseSpellsourceImageKey = (imageKey: string): ParsedImageKey => {
  const slashIdx = imageKey.lastIndexOf('/');
  const promptName = slashIdx >= 0 ? imageKey.slice(0, slashIdx) : imageKey;
  const arm = slashIdx >= 0 ? imageKey.slice(slashIdx + 1) : '';
  const m = promptName.match(/^(.*?)(?:_(256|512|1024))?(?:__s(\d+))?$/);
  const stem = m ? m[1] : promptName;
  const size = m && m[2] ? `${m[2]}×${m[2]}` : null;
  const seed = m && m[3] !== undefined ? m[3] : null;
  const promptInfo = SPELLSOURCE_PROMPT_CATALOG[stem] || null;

  let stageNumber = 2;
  let stageLabel = arm || 'Sample';
  let snappedStatus = 'Unsnapped';
  let isSnapped = false;

  if (arm === 'base') {
    stageNumber = 1;
    stageLabel = '1. Epoch 0 (untrained base)';
    snappedStatus = 'Unsnapped (raw model output)';
  } else if (arm === 'lora') {
    stageNumber = 2;
    stageLabel = '2. Latest checkpoint (trained)';
    snappedStatus = 'Unsnapped (raw model output)';
  } else if (arm === 'lora_vae') {
    stageNumber = 2;
    stageLabel = '2b. Latest checkpoint + Tuned VAE';
    snappedStatus = 'Unsnapped (tuned pixel-art VAE decode)';
  } else if (arm === 'snapped' || arm.startsWith('snapped.')) {
    stageNumber = 3;
    const method = arm.startsWith('snapped.') ? arm.slice('snapped.'.length) : 'lattice';
    stageLabel = '3. Latest → Snapped (postprocessed)';
    snappedStatus = `Snapped (${method} + border background removed)`;
    isSnapped = true;
  }

  return {
    rawKey: imageKey,
    promptName,
    stem,
    canvasSize: size,
    seed,
    arm,
    stageNumber,
    stageLabel,
    snappedStatus,
    isSnapped,
    promptInfo,
  };
};

export interface ParsedRunMetadata {
  modelTitle: string;
  epochsLabel: string;
  scaleLabel: string | null;
  resolutionLabel: string | null;
  kindLabel: string | null;
  stepLabel: string | null;
}

export const parseRunMetadata = (
  displayName: string,
  tags?: Record<string, KeyValueEntity>,
): ParsedRunMetadata | null => {
  if (!tags) return null;
  const trial = tags['trial']?.value || displayName;
  const evalKind = tags['eval_kind']?.value;
  const epoch = tags['epoch']?.value;
  const step = tags['step']?.value;
  const scaleTag = tags['scale']?.value;
  const kindTag = tags['kind']?.value;

  if (!evalKind && !scaleTag && !kindTag) {
    return null;
  }

  const resMatch = trial.match(/-(s\d+)-(c\d+)/);
  const scaleLabel = resMatch ? `${resMatch[1]} (${resMatch[1].slice(1)}px/cell)` : scaleTag ? `s${scaleTag} (${scaleTag}px/cell)` : null;
  const resolutionLabel = resMatch
    ? `${resMatch[2]} (${resMatch[2].slice(1)}×${resMatch[2].slice(1)} train)`
    : evalKind === 'base'
      ? 'c1024 (eval)'
      : null;

  let epochsLabel = '';
  if (evalKind === 'base' || epoch === '0' || step === '0') {
    epochsLabel = 'Epoch 0 (untrained, 0 steps)';
  } else if (evalKind === 'progress') {
    epochsLabel = epoch ? `Epoch ${epoch} (ongoing, step ${step || '?'})` : `Ongoing (step ${step || '?'})`;
  } else if (evalKind === 'final') {
    if (epoch) {
      epochsLabel = `${epoch} epochs (step ${step || '?'})`;
    } else if (trial.includes('stage2')) {
      epochsLabel = `Stage 2 (${step || '2000'} steps)`;
    } else {
      epochsLabel = `Finished (step ${step || '?'})`;
    }
  } else if (epoch) {
    epochsLabel = `${epoch} epochs`;
  }

  const kindLabel =
    kindTag === 'full' ? 'Full fine-tune' : kindTag === 'lora' ? 'LoRA' : kindTag === 'base' ? 'Untrained base' : null;

  return {
    modelTitle: trial,
    epochsLabel,
    scaleLabel,
    resolutionLabel,
    kindLabel,
    stepLabel: step ? `step ${step}` : null,
  };
};

export const ImageGridRunHeader = ({
  displayName,
  color,
  params,
  tags,
  maxParamsWidth,
}: {
  displayName: string;
  color?: string;
  params: Record<string, { key: string; value: string | number }>;
  tags?: Record<string, KeyValueEntity>;
  maxParamsWidth?: number;
}) => {
  const { theme } = useDesignSystemTheme();
  const paramsText = formatRunParams(params);
  const meta = parseRunMetadata(displayName, tags);

  const badgeStyle = (bg: string, fg: string) => ({
    display: 'inline-block',
    padding: '1px 6px',
    borderRadius: 4,
    fontSize: 11,
    fontWeight: 600 as const,
    lineHeight: 1.4,
    backgroundColor: bg,
    color: fg,
    whiteSpace: 'nowrap' as const,
  });

  const tooltipLines = [
    meta ? `${meta.modelTitle} — ${meta.epochsLabel}` : displayName,
    meta && [meta.scaleLabel, meta.resolutionLabel, meta.kindLabel].filter(Boolean).join(' • '),
    paramsText,
  ].filter(Boolean);

  return (
    <Tooltip
      content={tooltipLines.join('\n')}
      componentId="mlflow.charts.image-plot.run-name-tooltip"
    >
      <div css={{ overflow: 'hidden', display: 'flex', flexDirection: 'column', gap: 3, paddingBottom: 2 }}>
        <div
          css={{
            whiteSpace: 'nowrap',
            display: 'inline-flex',
            alignItems: 'center',
            gap: theme.spacing.xs,
            fontWeight: 600,
          }}
        >
          <RunColorPill color={color} />
          <span css={{ overflow: 'hidden', textOverflow: 'ellipsis' }}>
            {meta ? meta.modelTitle : displayName}
          </span>
        </div>
        {meta && (
          <div
            css={{
              display: 'flex',
              flexWrap: 'wrap',
              gap: 4,
              maxWidth: maxParamsWidth,
            }}
          >
            {meta.epochsLabel && (
              <span
                css={badgeStyle(
                  meta.epochsLabel.startsWith('Epoch 0')
                    ? theme.colors.backgroundSecondary
                    : theme.colors.actionDefaultBackgroundPress,
                  theme.colors.textPrimary,
                )}
              >
                {meta.epochsLabel}
              </span>
            )}
            {meta.scaleLabel && (
              <span css={badgeStyle(theme.colors.backgroundSecondary, theme.colors.textSecondary)}>
                Scale: {meta.scaleLabel}
              </span>
            )}
            {meta.resolutionLabel && (
              <span css={badgeStyle(theme.colors.backgroundSecondary, theme.colors.textSecondary)}>
                Res: {meta.resolutionLabel}
              </span>
            )}
            {meta.kindLabel && (
              <span css={badgeStyle(theme.colors.backgroundSecondary, theme.colors.textSecondary)}>
                {meta.kindLabel}
              </span>
            )}
          </div>
        )}
        {paramsText && (
          <div
            css={{
              fontSize: theme.typography.fontSizeSm,
              color: theme.colors.textSecondary,
              whiteSpace: 'normal',
              lineHeight: 1.3,
              maxWidth: maxParamsWidth,
              overflow: 'hidden',
            }}
          >
            {paramsText}
          </div>
        )}
      </div>
    </Tooltip>
  );
};

type ImagePlotProps = {
  imageUrl: string;
  compressedImageUrl: string;
  imageSize?: number;
  maxImageSize?: number;
};

export const ImagePlot = ({ imageUrl, compressedImageUrl, imageSize, maxImageSize }: ImagePlotProps) => {
  const [previewVisible, setPreviewVisible] = useState(false);
  const { theme } = useDesignSystemTheme();

  const [imageLoading, setImageLoading] = useState(true);

  useEffect(() => {
    // Load the image in the memory (should reuse the same request) in order to get the loading state
    setImageLoading(true);
    const img = new window.Image();
    img.onload = () => setImageLoading(false);
    img.onerror = () => setImageLoading(false);
    img.src = compressedImageUrl;
    return () => {
      img.src = '';
    };
  }, [compressedImageUrl]);

  return (
    <div css={{ width: imageSize || '100%', height: imageSize || '100%', minHeight: 0 }}>
      <div css={{ display: 'contents' }}>
        {compressedImageUrl === undefined || imageLoading ? (
          <div
            css={{
              width: '100%',
              backgroundColor: theme.colors.backgroundSecondary,
              display: 'flex',
              aspectRatio: '1',
              justifyContent: 'center',
              alignItems: 'center',
            }}
          >
            <Spinner />
          </div>
        ) : (
          <div
            css={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              width: imageSize || '100%',
              aspectRatio: '1',
              maxWidth: maxImageSize,
              // Bound by height as well as width so a square image shrinks to fit a
              // short card rather than overflowing it.
              maxHeight: maxImageSize ?? '100%',
              backgroundColor: theme.colors.backgroundSecondary,
              '.rc-image': {
                cursor: 'pointer',
              },
            }}
          >
            <ImagePreviewGroup visible={previewVisible} onVisibleChange={setPreviewVisible}>
              <Image
                src={compressedImageUrl}
                preview={{ src: imageUrl }}
                style={{ maxWidth: maxImageSize || '100%', maxHeight: maxImageSize || '100%' }}
              />
            </ImagePreviewGroup>
          </div>
        )}
      </div>
    </div>
  );
};

export const ImagePlotWithHistory = ({
  metadataByStep,
  imageSize,
  step,
  runUuid,
}: {
  metadataByStep: Record<number, ImageEntity>;
  imageSize?: number;
  step: number;
  runUuid: string;
}) => {
  const { theme } = useDesignSystemTheme();

  if (metadataByStep[step] === undefined) {
    return (
      <div
        css={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          textAlign: 'center',
          width: imageSize,
          backgroundColor: theme.colors.backgroundSecondary,
          padding: theme.spacing.md,
          aspectRatio: '1',
        }}
      >
        <ImageIcon />
        <FormattedMessage
          defaultMessage="No image logged at this step"
          description="Experiment tracking > runs charts > charts > image plot with history > no image text"
        />
      </div>
    );
  }
  return (
    <ImagePlot
      imageUrl={getArtifactLocationUrl(metadataByStep[step].filepath, runUuid)}
      compressedImageUrl={getArtifactLocationUrl(metadataByStep[step].compressed_filepath, runUuid)}
      imageSize={imageSize}
    />
  );
};

export const EmptyImageGridPlot = () => {
  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'center',
        alignItems: 'center',
        height: '100%',
        width: '100%',
        fontSize: 16,
      }}
    >
      <Typography.Title css={{ marginTop: 16 }} color="secondary" level={3}>
        Compare logged images
      </Typography.Title>
      <Typography.Text css={{ marginBottom: 16 }} color="secondary">
        Use the image grid chart to compare logged images across runs.
      </Typography.Text>
    </div>
  );
};
