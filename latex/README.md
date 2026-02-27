# Local Setup and Rendering Guide for EMNLP LaTeX Template

This guide will walk you through setting up your local environment on Windows to compile the `paper.tex` file using the EMNLP/ACL style.

## 1. Prerequisites (Installing LaTeX)

You need a LaTeX distribution installed on your machine. On Windows, the most popular and easiest to use is **MiKTeX**.

1. Go to the [MiKTeX Download Page](https://miktex.org/download).
2. Download the basic installer.
3. Run the installer and follow the instructions. During installation, it's recommended to set "Install missing packages on-the-fly" to **Yes**.

*Alternative:* You can install **TeX Live**, which is much larger but includes almost all LaTeX packages out of the box.

## 2. Editor Recommendations

While you can compile from the command line, using a dedicated editor makes it much easier:

*   **VS Code**: Highly recommended. 
    *   Install the **LaTeX Workshop** extension by James Yu. It provides syntax highlighting, auto-completion, and PDF viewer.
*   **TeXstudio**: A dedicated LaTeX IDE (comes with its own built-in PDF viewer and compiling tools).

## 3. Required EMNLP/ACL Style Files

The `paper.tex` uses `\usepackage[review]{acl}` and `\bibliographystyle{acl_natbib}`. To compile it successfully, you MUST have the official ACL style files in the same directory as your `paper.tex` file.

1. Download the latest ACL/EMNLP template (e.g., from [EMNLP Author Guidelines](https://2023.emnlp.org/calls/style-and-formatting/)).
2. Extract the archive.
3. Copy the following two files into this `latex` directory:
    *   `acl.sty`
    *   `acl_natbib.bst`

## 4. Compiling the Document

### Method A: Using VS Code (Recommended)
1. Open the `latex` folder in VS Code.
2. Open `paper.tex`.
3. If you have LaTeX Workshop installed, a "Build" button (looks like a play button) will appear in the top right. Click it. 
    * *Or use the shortcut*: `Ctrl+Alt+B`.
4. The extension will automatically handle running `pdflatex` -> `bibtex` -> `pdflatex` -> `pdflatex` to generate the PDF and resolve citations.
5. Click the "View PDF" button (top right) or `Ctrl+Alt+V` to see the compiled result.

### Method B: Using Command Line (PowerShell/CMD)
LaTeX requires multiple passes to properly link citations and references. Open your terminal in this `latex` directory and run the following commands sequentially:

```powershell
# 1. First pass to generate auxiliary files (.aux)
pdflatex paper.tex

# 2. Run bibtex to process the custom.bib file using the .aux file
bibtex paper.aux

# 3. Second pass to include the compiled bibliography into the document
pdflatex paper.tex

# 4. Final pass to resolve all cross-references correctly
pdflatex paper.tex
```

This will generate a `paper.pdf` file in the directory.

## Troubleshooting

*   **"File `acl.sty` not found"**: You forgot to download the ACL style file and place it in the directory. Follow Step 3.
*   **Missing Package Errors**: If MiKTeX asks to install missing packages (like `times`, `microtype`, `inconsolata`, etc.), grant it permission to do so.
*   **Question mark `[?]` instead of citations**: This means `bibtex` hasn't run properly or `pdflatex` needs another run to update the references. Follow the 4 compile steps in Method B.
