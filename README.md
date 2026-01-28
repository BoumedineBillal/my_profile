# My Web Profile

This is the repository for my personal web profile. It is built with a scalable, standard directory structure suitable for modern web development, designed to be hosted on GitHub Pages.

## Project Structure

The project follows a component-based and modular architecture:

```text
my_profile/
├── index.html              # Main entry point
├── README.md               # Project documentation
├── .gitignore              # Git ignore rules
├── public/                 # Static assets (favicons, robots.txt, etc.)
│   └── images/             # Static images
└── src/                    # Source code
    ├── assets/             # processed assets (fonts, icons)
    ├── css/                # Styling
    │   ├── main.css        # Main stylesheet (imports others)
    │   ├── variables.css   # CSS variables (colors, fonts, spacing)
    │   ├── layout.css      # Structural styles (header, footer, grid)
    │   └── components.css  # Reusable UI components (buttons, cards)
    └── js/                 # JavaScript logic
        ├── main.js         # Main script (entry point)
        └── modules/        # Modular JS code
            └── greetings.js # Example module
```

## Getting Started

1.  **Clone the repository**.
2.  **Open `index.html`** in your web browser.
3.  **Edit**: Modify files in `src/` to update the content and styles.

## CSS Architecture

- **`variables.css`**: Define your design tokens here (colors, spacing, fonts). Changing a value here updates the entire site.
- **`layout.css`**: Handles the big picture—how the page is divided.
- **`components.css`**: Styles for individual elements like buttons ("Components").

## JavaScript

The project uses ES6 Modules. `main.js` imports functionality from `src/js/modules/`, keeping the code clean and separated.
