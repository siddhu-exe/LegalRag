---
name: LegalRAG Showcase
colors:
  surface: '#10131a'
  surface-dim: '#10131a'
  surface-bright: '#363940'
  surface-container-lowest: '#0b0e14'
  surface-container-low: '#191c22'
  surface-container: '#1d2026'
  surface-container-high: '#272a31'
  surface-container-highest: '#32353c'
  on-surface: '#e1e2eb'
  on-surface-variant: '#d0c5af'
  inverse-surface: '#e1e2eb'
  inverse-on-surface: '#2e3037'
  outline: '#99907c'
  outline-variant: '#4d4635'
  surface-tint: '#e9c349'
  primary: '#f2ca50'
  on-primary: '#3c2f00'
  primary-container: '#d4af37'
  on-primary-container: '#554300'
  inverse-primary: '#735c00'
  secondary: '#aec9e7'
  on-secondary: '#16324a'
  secondary-container: '#2e4962'
  on-secondary-container: '#9db8d5'
  tertiary: '#48e5d0'
  on-tertiary: '#003731'
  tertiary-container: '#11c9b4'
  on-tertiary-container: '#004f46'
  error: '#ffb4ab'
  on-error: '#690005'
  error-container: '#93000a'
  on-error-container: '#ffdad6'
  primary-fixed: '#ffe088'
  primary-fixed-dim: '#e9c349'
  on-primary-fixed: '#241a00'
  on-primary-fixed-variant: '#574500'
  secondary-fixed: '#cee5ff'
  secondary-fixed-dim: '#aec9e7'
  on-secondary-fixed: '#001d33'
  on-secondary-fixed-variant: '#2e4962'
  tertiary-fixed: '#62fae3'
  tertiary-fixed-dim: '#3cddc7'
  on-tertiary-fixed: '#00201c'
  on-tertiary-fixed-variant: '#005047'
  background: '#10131a'
  on-background: '#e1e2eb'
  surface-variant: '#32353c'
typography:
  display-lg:
    fontFamily: Newsreader
    fontSize: 56px
    fontWeight: '400'
    lineHeight: 64px
    letterSpacing: -0.02em
  display-lg-mobile:
    fontFamily: Newsreader
    fontSize: 36px
    fontWeight: '400'
    lineHeight: 44px
    letterSpacing: -0.01em
  headline-lg:
    fontFamily: Newsreader
    fontSize: 40px
    fontWeight: '400'
    lineHeight: 48px
    letterSpacing: -0.015em
  headline-lg-mobile:
    fontFamily: Newsreader
    fontSize: 28px
    fontWeight: '400'
    lineHeight: 36px
    letterSpacing: -0.01em
  headline-md:
    fontFamily: Newsreader
    fontSize: 28px
    fontWeight: '400'
    lineHeight: 36px
    letterSpacing: -0.01em
  headline-sm:
    fontFamily: Inter
    fontSize: 20px
    fontWeight: '600'
    lineHeight: 28px
    letterSpacing: -0.005em
  body-lg:
    fontFamily: Inter
    fontSize: 17px
    fontWeight: '400'
    lineHeight: 28px
    letterSpacing: -0.002em
  body-md:
    fontFamily: Inter
    fontSize: 15px
    fontWeight: '400'
    lineHeight: 24px
  body-sm:
    fontFamily: Inter
    fontSize: 13px
    fontWeight: '400'
    lineHeight: 20px
  citation-quote:
    fontFamily: Newsreader
    fontSize: 18px
    fontWeight: '400'
    lineHeight: 28px
    letterSpacing: 0.01em
  code-sm:
    fontFamily: JetBrains Mono
    fontSize: 12px
    fontWeight: '400'
    lineHeight: 18px
  label-mono:
    fontFamily: JetBrains Mono
    fontSize: 11px
    fontWeight: '500'
    lineHeight: 16px
    letterSpacing: 0.06em
  label-ui:
    fontFamily: Inter
    fontSize: 12px
    fontWeight: '500'
    lineHeight: 16px
    letterSpacing: 0.02em
rounded:
  sm: 0.125rem
  DEFAULT: 0.25rem
  md: 0.375rem
  lg: 0.5rem
  xl: 0.75rem
  full: 9999px
spacing:
  gutter: 1.5rem
  gutter-sm: 1rem
  gutter-lg: 2rem
  margin: 2rem
  margin-sm: 1.25rem
  margin-lg: 4rem
  space-xs: 0.25rem
  space-sm: 0.5rem
  space-md: 1rem
  space-lg: 1.5rem
  space-xl: 2.5rem
---

## Brand & Style

This design system embodies the tension between high-jurisprudence scholarship and bleeding-edge autonomous intelligence. It is crafted for elite legal-tech engineering, targeting general counsels, legal scholars, institutional partners, and engineering directors.

The aesthetic fuses **Editorial Minimalism** with **Technical Precision**:
- **Atmosphere:** Deep obsidian, midnight slate, and muted parchment tones evoking bespoke private chambers and specialized research labs.
- **Craft Details:** Razor-sharp 1px hairline dividers, whisper-soft backdrop blurs, and strict spatial alignments that reject generic venture-backed SaaS tropes.
- **Distinction:** Academic authority balanced with technical rigor. Code snippets, latency metrics, and cosine similarity values sit naturally alongside Supreme Court citations and statutory cross-references.

## Colors

The palette operates on a nocturnal spectrum with high-value functional accents:

- **Neutral Foundation (`#0B0E14`):** A profound midnight obsidian serving as the base canvas, paired with structural tiers of `#111622` (surface-base), `#161D2B` (surface-raised), and `#1E2638` (surface-overlay).
- **Primary Accent (`#D4AF37` - Legal Brass/Amber):** Reserved strictly for evidentiary verification, authority badges, verified citational seals, and active primary focus states. It acts as a stamp of judicial authenticity.
- **Secondary Accent (`#7B96B2` - Slate Steel):** Governs structural metadata, vector dimensions, retrieval graph links, and secondary interactive iconography.
- **Tertiary Accent (`#2DD4BF` - Vector Mint):** Denotes precision telemetry—sub-millisecond retrieval latencies, high cosine similarities (>0.92), and live cluster states.
- **Hairline Borders (`rgba(255, 255, 255, 0.08)`): Used uniformly for 1px structural framing and container boundaries.

## Typography

The typographic hierarchy establishes dual registers of humanistic legal tradition and algorithmic rigor:

- **Editorial Serifs (`Newsreader`):** Utilized for section landmarks, case titles, thesis statements, and statutory pull quotes. It brings warmth, intellectual gravity, and editorial restraint. Italic styles are strictly reserved for legal case citations (e.g., *Chevron U.S.A., Inc. v. Natural Resources Defense Council, Inc.*).
- **System Interface (`Inter`):** Powers body prose, complex analytical breakdowns, and dense document analysis interfaces to guarantee cross-screen legibility and high information density.
- **Telemetry & Metadata (`JetBrains Mono`):** Dedicated to legal citation numbers (e.g., `467 U.S. 837 (1984)`), query latency, embedding chunk indices, token counts, and cosine similarity markers. All labels rendered in monospace must feature uppercase capitalization with explicit letter-spacing.

## Layout & Spacing

The layout is governed by a 12-column asymmetric fluid grid, engineered for deep dual-pane interfaces (contextual retrieval inspector alongside statutory document reading panes):

- **Grid Architecture:** 
  - **Desktop (≥1280px):** 12 columns, 2rem (`gutter-lg`) gutters, 4rem (`margin-lg`) external margins. Content caps at 1440px max-width to protect editorial reading lines.
  - **Tablet (768px - 1279px):** 8 columns, 1.5rem (`gutter`) gutters, 2rem (`margin`) margins. Side-by-side verification panels stack or convert into an overlay drawer.
  - **Mobile (<768px):** 4 columns, 1rem (`gutter-sm`) gutters, 1.25rem (`margin-sm`) margins. Primary research inputs dock along the bottom viewport.
- **Rhythmic Rules:** Component paddings use `space-md` (1rem) for condensed analytical chips and `space-xl` (2.5rem) for major editorial content segments. Hairline dividers act as structural boundaries rather than pure visual spacers.

## Elevation & Depth

This design system avoids heavy drop shadows, instead creating spatial hierarchy through surface luminance, subtle frosted glassmorphism, and structural 1px ghost borders:

- **Level 0 (Base Canvas):** Background tone `#0B0E14` with an optional 0.3% faint stochastic noise layer to soften screen emission.
- **Level 1 (Panels & Deep Containers):** Solid `#111622` surfaced with a crisp 1px border of `rgba(255, 255, 255, 0.07)`. No shadow.
- **Level 2 (Active Cards & Citations):** Surface `#161D2B` with a subtle inner top-highlight (`inset 0 1px 0 0 rgba(255, 255, 255, 0.08)`) and border `rgba(255, 255, 255, 0.12)`. Ambient shadow: `0 8px 32px -4px rgba(0, 0, 0, 0.6)`.
- **Level 3 (Inspection Modals & Flyouts):** Frosted glass canvas using `rgba(17, 22, 34, 0.75)` with `backdrop-filter: blur(16px)` and a hairline border of `rgba(212, 175, 55, 0.25)` (amber glow) when highlighting verified evidence.
- **Evidentiary Glow:** Elements carrying verified legal grounding emit a subtle, diffuse accent aura: `0 0 20px -4px rgba(212, 175, 55, 0.15)`.

## Shapes

To reinforce architectural permanence and academic authority, shapes are disciplined, crisp, and minimally rounded (`roundedness: 1`):

- **Micro Components (Chips, Badges, Metrics):** 2px to 4px (`rounded-sm`).
- **Standard UI (Buttons, Search Bars, Inputs):** 4px (`0.25rem`).
- **Structural Containers (Dockets, Document Viewers, Cards):** 8px (`0.5rem`).
- **Pills & Over-rounded elements are strictly prohibited**, with the single exception of high-priority live node indicators (e.g., circular ping lights). Sharp corners are preferred over playful or casual curves.

## Components

### Buttons
- **Primary (Action/Run Retrieval):** Background `#D4AF37`, foreground `#0B0E14` (weight: 600). Hover state subtly brightens to `#E5C358` with a 1px border `rgba(255, 255, 255, 0.2)`. Active state compresses slightly without shadow.
- **Secondary (Inspect Nodes/Tokens):** Background `rgba(255, 255, 255, 0.03)`, text `#FFFFFF`, border `1px solid rgba(255, 255, 255, 0.1)`. Hover shifts background to `rgba(255, 255, 255, 0.06)` and text to `#D4AF37`.
- **Ghost/Tertiary:** Zero background, text `#7B96B2`, hover color `#FFFFFF` with an understated text underline.

### Verification Chips & Metadata Tags
- **Statute Verification Chip:** Monospaced label (`JetBrains Mono`), 1px solid `rgba(212, 175, 55, 0.3)`, background `rgba(212, 175, 55, 0.06)`, text `#D4AF37`. Features an optional leading dot indicator with a micro pulse animation.
- **Latency / Telemetry Chip:** Monospaced label, background `rgba(45, 212, 191, 0.06)`, text `#2DD4BF`, border `1px solid rgba(45, 212, 191, 0.2)`.

### Cards & Docket Containers
- Framed in 1px borders `rgba(255, 255, 255, 0.08)`.
- Card headers utilize `Newsreader` for case titles and `JetBrains Mono` for docket indices, separated by a crisp 1px rule.
- Hovering an interactive docket card triggers a transition of the border from `rgba(255, 255, 255, 0.08)` to `rgba(212, 175, 55, 0.4)` over 200ms.

### Form Inputs & Search Command Bars
- Dark recessed background `#070A0F`, 1px border `rgba(255, 255, 255, 0.12)`.
- Typography is set to `Inter` 15px for query syntax, with an inline keyboard shortcut badge in `JetBrains Mono` (`⌘K`).
- Focus state switches the border to `#D4AF37` with an ambient glow of `0 0 0 1px #D4AF37`.

### Legal Citation Accordion & Chunk Inspector
- Displays extracted corpus context chunks with source statutory numbers.
- Highlighting over cited phrases reveals a contextual citation tooltip using `Newsreader` italic, displaying the exact jurisdictional paragraph, court, and retrieval confidence score.