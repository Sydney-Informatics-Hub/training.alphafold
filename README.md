# SIH AlphaFold2 Training

## File location

- Store `index.qmd`, `setup.qmd` files in the root directory 
- Store notebooks (.ipynb), R files (as .qmd or .rmd), and markdown lessons (.qmd or .md) in the `notebooks` folder
- Store figures in a `figs` folder 

## How to use this repo 

1. Edit `index.qmd` to change the main landing page. This is basically a markdown file.
2. Edit or create `setup.qmd` to change the Setup instruction pages. Same - basically a md file.
3. Edit `_quarto.yml` to change the dropdown menu options.
4. Add additional `*.md` files to the root dir to have them converted to html files (and add them to `_quarto.yml` to make them navigable), if you'd like.
5. Edit this Readme in your fork to reflect the content of your workshop.

The project will be built and rendered automatically (via github actions) at a URL with this format:
```html
https://pages.github.sydney.edu.au/informatics/training.alphafold/
```

### Themes, Aesthetic and Branding

If you'd like to use a more generic and possibly neutral theme, go to the `_quarto.yaml` and change the format section to:

```yaml
format:
  html:
   toc: true
   theme:
      light: flatly
      dark: darkly
   css: styles.scss
   code-link: true
   code-fold: false
```

If you'd like to use the USYD Masterbrand Ochre, go to the `_quarto.yaml` and change the format section to:

```yaml
format:
  html:
    theme: simplex
    css: [lesson.css, bootstrap-icons.css]
    toc: true
    code-overflow: wrap
    highlight-style: github
```
 

