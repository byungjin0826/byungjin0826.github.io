// @ts-check
const { fontFamily } = require('tailwindcss/defaultTheme')
const colors = require('tailwindcss/colors')

// 시트러스(라임) 스케일 — 레거시 primary 유틸리티가 이 팔레트를 쓰도록 매핑
const citrus = {
  50: '#f6fbe8',
  100: '#eef8d4',
  200: '#d4f59a',
  300: '#b5f13f',
  400: '#a3e635',
  500: '#5b8c00',
  600: '#4f7a00',
  700: '#3f6100',
  800: '#33500a',
  900: '#2b4200',
  950: '#1c2c00',
}

/** @type {import("tailwindcss/types").Config } */
module.exports = {
  content: [
    './node_modules/pliny/**/*.js',
    './app/**/*.{js,ts,jsx,tsx}',
    './pages/**/*.{js,ts,tsx}',
    './components/**/*.{js,ts,tsx}',
    './layouts/**/*.{js,ts,tsx}',
    './data/**/*.mdx',
    './data/**/*.md',
  ],
  darkMode: 'class',
  theme: {
    extend: {
      lineHeight: {
        11: '2.75rem',
        12: '3rem',
        13: '3.25rem',
        14: '3.5rem',
      },
      fontFamily: {
        display: ['Archivo', 'Pretendard', ...fontFamily.sans],
        sans: ['Pretendard', 'Archivo', ...fontFamily.sans],
        mono: ['JetBrains Mono', ...fontFamily.mono],
      },
      maxWidth: {
        container: 'var(--container)',
        prose: 'var(--container-prose)',
      },
      colors: {
        primary: citrus,
        citrus,
        gray: colors.gray,
        // 시맨틱 토큰 (CSS 변수 기반 · 테마 반응형)
        page: 'var(--bg)',
        surface: {
          DEFAULT: 'var(--surface)',
          2: 'var(--surface-2)',
          sunken: 'var(--surface-sunken)',
        },
        ink: {
          DEFAULT: 'var(--text)',
          muted: 'var(--text-muted)',
          subtle: 'var(--text-subtle)',
          inverse: 'var(--text-inverse)',
        },
        line: {
          DEFAULT: 'var(--border)',
          strong: 'var(--border-strong)',
        },
        rule: 'var(--rule)',
        accent: {
          DEFAULT: 'var(--accent)',
          hover: 'var(--accent-hover)',
          press: 'var(--accent-press)',
          text: 'var(--accent-text)',
          soft: 'var(--accent-soft)',
        },
        'on-accent': 'var(--on-accent)',
        info: 'var(--info)',
        warning: 'var(--warning)',
        danger: 'var(--danger)',
        success: 'var(--success)',
      },
      borderRadius: {
        sm: 'var(--radius-sm)',
        md: 'var(--radius-md)',
        lg: 'var(--radius-lg)',
        xl: 'var(--radius-xl)',
      },
      keyframes: {
        'fade-up': {
          '0%': { opacity: '0', transform: 'translateY(6px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
      },
      animation: {
        'fade-up': 'fade-up var(--dur-base) var(--ease-out) both',
      },
      typography: () => ({
        DEFAULT: {
          css: {
            '--tw-prose-body': 'var(--text)',
            '--tw-prose-headings': 'var(--text)',
            '--tw-prose-lead': 'var(--text-muted)',
            '--tw-prose-links': 'var(--accent-text)',
            '--tw-prose-bold': 'var(--text)',
            '--tw-prose-counters': 'var(--text-muted)',
            '--tw-prose-bullets': 'var(--text-subtle)',
            '--tw-prose-hr': 'var(--border)',
            '--tw-prose-quotes': 'var(--text)',
            '--tw-prose-quote-borders': 'var(--accent)',
            '--tw-prose-captions': 'var(--text-muted)',
            '--tw-prose-code': 'var(--text)',
            '--tw-prose-pre-code': 'var(--text)',
            '--tw-prose-pre-bg': 'var(--surface-2)',
            '--tw-prose-th-borders': 'var(--border)',
            '--tw-prose-td-borders': 'var(--border)',
            maxWidth: 'none',
            fontFamily: 'var(--font-sans)',
            fontSize: '1.0625rem',
            lineHeight: '1.85',
            a: {
              fontWeight: '500',
              textUnderlineOffset: '3px',
            },
            'h1,h2,h3,h4': {
              fontFamily: 'var(--font-sans)',
              fontWeight: '800',
              letterSpacing: '-0.02em',
            },
            code: {
              fontFamily: 'var(--font-mono)',
              fontWeight: '500',
              backgroundColor: 'var(--surface-2)',
              padding: '0.15em 0.4em',
              borderRadius: '4px',
            },
            'code::before': { content: '""' },
            'code::after': { content: '""' },
            blockquote: {
              fontStyle: 'normal',
              fontWeight: '500',
              borderLeftWidth: '3px',
            },
          },
        },
        invert: {
          css: {
            '--tw-prose-body': 'var(--text)',
            '--tw-prose-headings': 'var(--text)',
            '--tw-prose-links': 'var(--accent-text)',
            '--tw-prose-bold': 'var(--text)',
            '--tw-prose-quotes': 'var(--text)',
            '--tw-prose-quote-borders': 'var(--accent)',
            '--tw-prose-code': 'var(--text)',
            '--tw-prose-pre-bg': 'var(--surface-2)',
            '--tw-prose-hr': 'var(--border)',
          },
        },
      }),
    },
  },
  plugins: [require('@tailwindcss/forms'), require('@tailwindcss/typography')],
}
