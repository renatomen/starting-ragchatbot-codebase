# Frontend Changes: Theme Toggle Button & Light Theme

## Overview

Added a theme toggle button that allows users to switch between dark and light modes. The toggle is positioned in the top-right corner and uses sun/moon icons with smooth animations. Implemented a comprehensive light theme with proper contrast ratios meeting WCAG accessibility standards.

## Files Modified

### 1. `frontend/index.html`

Added the theme toggle button element after the `<body>` tag:
- Circular button with `id="themeToggle"` and `class="theme-toggle"`
- Contains two SVG icons: sun icon (for dark mode) and moon icon (for light mode)
- Includes accessibility attributes: `aria-label`, `title`, and `type="button"`

### 2. `frontend/style.css`

#### Light Theme CSS Variables

Added `[data-theme="light"]` selector with comprehensive color palette:

| Variable | Light Theme Value | Purpose |
|----------|-------------------|---------|
| `--primary-color` | `#1d4ed8` | Darker blue for better contrast on light bg |
| `--primary-hover` | `#1e40af` | Even darker blue for hover states |
| `--background` | `#f8fafc` | Light gray page background |
| `--surface` | `#ffffff` | White surface for cards/panels |
| `--surface-hover` | `#f1f5f9` | Subtle hover state |
| `--text-primary` | `#0f172a` | Near-black for maximum readability |
| `--text-secondary` | `#475569` | Dark gray for secondary text |
| `--border-color` | `#cbd5e1` | Visible but subtle borders |
| `--user-message` | `#1d4ed8` | User message bubble color |
| `--assistant-message` | `#f1f5f9` | Light gray for assistant messages |
| `--shadow` | Lighter opacity | Subtle shadows for light theme |
| `--code-bg` | `#f1f5f9` | Light gray code background |
| `--source-tag-bg` | `rgba(29, 78, 216, 0.1)` | Subtle blue tint |
| `--source-tag-border` | `rgba(29, 78, 216, 0.3)` | Blue border |
| `--source-tag-text` | `#1d4ed8` | Blue text |
| `--error-bg/border/text` | Red variants | Error message styling |
| `--success-bg/border/text` | Green variants | Success message styling |

#### Theme Toggle Button Styles

- Fixed position in top-right corner (`top: 1rem`, `right: 1rem`)
- Circular 44x44px button with border and shadow
- Hover state with scale transform and border color change
- Focus state with focus ring for accessibility
- Active state with press-down animation

#### Icon Animation Styles

- Sun and moon icons use CSS transitions for smooth switching
- Icons rotate and scale during transition (0.3s ease)
- In dark mode: sun icon visible, moon icon hidden
- In light mode: moon icon visible, sun icon hidden

#### Smooth Theme Transitions

Added transitions to multiple elements for seamless theme switching:
- `body`: background-color, color
- `.sidebar`: background-color, border-color
- `.message-content`: background-color, color
- `.chat-input-container`: background-color, border-color
- `#chatInput`: background-color, border-color, color

### 3. `frontend/script.js`

**New Theme Management Functions:**

```javascript
initializeTheme()  // Checks localStorage and system preference
setTheme(theme)    // Applies theme and updates aria-label
toggleTheme()      // Switches between light and dark
```

**Features:**
- Theme preference persisted to `localStorage`
- Respects system preference (`prefers-color-scheme`) on first visit
- Updates `aria-label` dynamically for screen readers
- Keyboard support (Enter and Space keys)

## Accessibility Features

### WCAG Compliance

1. **Color Contrast Ratios:**
   - Primary text (`#0f172a` on `#f8fafc`): ~15:1 ratio (exceeds AAA)
   - Secondary text (`#475569` on `#f8fafc`): ~7:1 ratio (meets AAA)
   - Primary color (`#1d4ed8` on white): ~4.8:1 ratio (meets AA for large text)

2. **Keyboard Navigation:**
   - Button is focusable and can be activated with Enter or Space
   - Clear focus ring indicator using `--focus-ring` color
   - Tab order maintains logical flow

3. **Screen Reader Support:**
   - `aria-label` updates dynamically based on current theme
   - `title` attribute provides tooltip on hover
   - Button role is implicit from `<button>` element

4. **Touch Targets:**
   - 44x44px minimum touch target size (meets WCAG 2.5.5)

## Color Choices Rationale

### Light Theme Design Decisions

1. **Background Colors:**
   - `#f8fafc` chosen over pure white to reduce eye strain
   - Provides subtle distinction between background and surface

2. **Text Colors:**
   - `#0f172a` for primary text ensures maximum readability
   - `#475569` for secondary text maintains hierarchy while staying readable

3. **Primary Blue:**
   - Darkened from `#2563eb` to `#1d4ed8` for better contrast on light backgrounds
   - Maintains brand consistency while meeting accessibility standards

4. **Borders:**
   - `#cbd5e1` provides visible structure without being harsh
   - Slightly darker than light theme surface colors for definition

## Usage

The toggle button appears in the top-right corner of the page. Users can:
- Click the button to toggle between themes
- Use keyboard Tab to focus and Enter/Space to activate
- Theme preference is automatically saved and restored on page reload
- System preference is respected on first visit
