import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ["var(--font-outfit)"],
        mono: ["var(--font-mono)"],
      },
      colors: {
        background: "var(--background)",
        foreground: "var(--foreground)",
        slate: {
          950: "#020617",
          900: "#0f172a",
          800: "#1e293b",
        },
        cyan: {
          500: "#06b6d4",
          400: "#22d3ee",
        }
      },
      animation: {
        "accordion-down": "accordion-down 0.2s ease-out",
        "accordion-up": "accordion-up 0.2s ease-out",
      },
    },
  },
  plugins: [],
};
export default config;
