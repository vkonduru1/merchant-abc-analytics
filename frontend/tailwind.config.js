/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        brand: {
          navy:   '#17274C',
          orange: '#C85510',
          paper:  '#FBF8F2',
        }
      }
    },
  },
  plugins: [],
}
