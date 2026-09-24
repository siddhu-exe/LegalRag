/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        canvas: {
          base: "#0B0E14",
          surface: "#10131a",
          surfaceLow: "#191c22",
          surfaceContainer: "#1d2026",
          surfaceHigh: "#272a31",
          surfaceHighest: "#32353c"
        },
        brass: {
          DEFAULT: "#D4AF37",
          light: "#f2ca50",
          dim: "#e9c349",
          dark: "#554300"
        },
        slateSteel: {
          DEFAULT: "#7B96B2",
          light: "#aec9e7",
          dark: "#2e4962"
        },
        vectorMint: {
          DEFAULT: "#2DD4BF",
          light: "#48e5d0",
          dim: "#11c9b4",
          dark: "#004f46"
        }
      },
      fontFamily: {
        serif: ["Newsreader", "Georgia", "serif"],
        sans: ["Inter", "-apple-system", "BlinkMacSystemFont", "sans-serif"],
        mono: ["JetBrains Mono", "Menlo", "monospace"]
      },
      keyframes: {
        'fade-in': {
          from: { opacity: '0', transform: 'translateY(-2px)' },
          to: { opacity: '1', transform: 'translateY(0)' }
        }
      },
      animation: {
        'fade-in': 'fade-in 200ms ease-out'
      }
    },
  },
  plugins: [],
}
