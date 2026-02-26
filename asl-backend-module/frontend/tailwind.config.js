export default {
    content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
    theme: {
        extend: {
            colors: {
                primary: '#667eea',
                secondary: '#764ba2',
                success: '#06ffa5',
                warning: '#ffc400',
                danger: '#ff3838',
            },
            fontFamily: {
                sans: ['Inter', 'sans-serif'],
            },
            spacing: {
                '128': '32rem',
                '144': '36rem',
            },
            animation: {
                'fade-in-up': 'fadeInUp 0.5s ease-out',
            },
        },
    },
    plugins: [],
};